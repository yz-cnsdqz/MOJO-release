import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
sys.path.append(os.getcwd())
from utils import *
from experiments.utils.config import Config
from experiments.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized


def loss_function(X, Y_r, Y, mu, logvar):
    MSE = F.l1_loss(Y_r, Y) + cfg.lambda_tf*F.l1_loss(Y_r[1:]-Y_r[:-1], Y[1:]-Y[:-1])
    MSE_v = F.l1_loss(X[-1], Y_r[0])
    KLD = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())
    if robustkl:
        KLD = torch.sqrt(1 + KLD**2)-1
    loss_r = MSE + cfg.lambda_v * MSE_v + cfg.beta * KLD
    return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v.item(), KLD.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'MSE', 'MSE_v', 'KLD']
    while batch_gen.has_next_rec():
        traj = batch_gen.next_batch(cfg.batch_size).to(device)
        if (torch.isnan(traj)).any():
            print('- meet nan. Skip it')
            continue
        X = traj[:t_his]
        Y = traj[t_his:]
        Y_r, mu, logvar = model(X, Y)
        loss, losses = loss_function(X, Y_r, Y, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        total_num_sample += 1
    batch_gen.reset()

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalar('vae_' + name, loss, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    """load the right model"""
    if 'vanilla' in args.cfg:
        from models.models_vanilla import *
    elif 'mojo' in args.cfg:
        from models.models_mojo import *

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    body_repr = cfg.body_repr
    subsets = cfg.dataset
    robustkl = cfg.robustkl

    """data"""
    batch_gen = BatchGeneratorAMASSCanonicalized(amass_data_path=cfg.dataset_path,
                                                 amass_subset_name=subsets,
                                                 sample_rate=8,
                                                 body_repr=body_repr)
    batch_gen.get_rec_list()

    """model"""
    model = get_vae_model(cfg, batch_gen.get_feature_dim())
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path)
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict()}
                    torch.save(model_cp, cp_path)



