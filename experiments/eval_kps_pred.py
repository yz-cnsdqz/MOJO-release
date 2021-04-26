'''
this script is to evaluate stochastic motion prediction on amass
'''
import numpy as np
import argparse
import os
import sys
import pickle
import csv

sys.path.append(os.getcwd())
from utils import *
from experiments.utils.config import Config
from experiments.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from experiments.utils.eval_metrics import *


def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True):
    traj_np = data
    if type(traj_np) is np.ndarray:
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    elif type(traj_np) is torch.Tensor:
        traj = traj_np.permute(1,0,2).clone().detach()
    X = traj[:t_his] # original setting

    if algo == 'dlow':
        X = X.repeat((1, num_seeds, 1))
        Z_g = models[algo].sample(X)
        if 'mojo' in args.cfg:
            Z_highfreq = torch.randn((t_pred-n_freq, Z_g.shape[1], Z_g.shape[2]), device=Z_g.device)
            Z_g = torch.cat([Z_g, Z_highfreq],dim=0) #[freq, b*nk, d]
        X = X.repeat_interleave(sample_num, dim=1)
        Y = models['vae'].decode(X, Z_g)

    elif algo == 'vae':
        X = X.repeat((1, sample_num * num_seeds, 1))
        Y = models[algo].sample_prior(X)

    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
    Y = Y.permute(1, 0, 2).detach().cpu().numpy()

    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...] #expand a dim
    return Y




def visualize(n_seqs, n_gens):
    '''
    Actually this is just to generate files for visualization, rather than directly render them.
    n_seqs: how many sequences to generate
    n_gens: for each input sequence, how many different sequences to predict
    '''
    ### generate data and save them to files. They will need inverse kinematics to get body mesh.
    ### generate data
    gen_results = {}
    gen_results['gt'] = []
    gen_results['betas'] = []
    gen_results['gender'] = []
    gen_results['transf_rotmat'] = []
    gen_results['transf_transl'] = []
    for algo in vis_algos:
        gen_results[algo] = []
    idx = 0
    while idx < n_seqs:
        data = batch_gen.next_sequence()
        motion_np = data['body_feature']
        motion = torch.FloatTensor(motion_np).unsqueeze(0) #[b,t,d]
        gen_results['gt'].append(motion_np.reshape((1,motion_np.shape[0],-1,3)))
        gen_results['betas'].append(data['betas'])
        gen_results['gender'].append(str(data['gender']))
        gen_results['transf_rotmat'].append(data['transf_rotmat'])
        gen_results['transf_transl'].append(data['transf_transl'])
        # vae
        for algo in vis_algos:
            pred = get_prediction(motion, algo, cfg.nk)[0]
            pred = np.reshape(pred, (pred.shape[0], pred.shape[1],-1,3))
            pred = pred[:n_gens]
            gen_results[algo].append(pred)
        idx+=1
    gen_results['gt'] = np.stack(gen_results['gt'])
    for algo in vis_algos:
        gen_results[algo] = np.stack(gen_results[algo]) #[#seq, #genseq_per_pastmotion, t, #joints, 3]

    ### save to file
    outfilename = '{}/seq_gen_seed{}/{}/'.format(cfg.result_dir, args.num_seeds, testing_data[0])
    if not os.path.exists(outfilename):
        os.makedirs(outfilename)
    outfilename += 'results_{}.pkl'.format(body_repr)
    with open(outfilename, 'wb') as f:
        pickle.dump(gen_results, f)



def compute_stats():
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde,
                  'FE': compute_ps_entropy}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    num_samples = 0
    num_seeds = args.num_seeds
    for i in range(all_data.shape[0]):
        data = all_data[i:i+1]
        num_samples += 1
        gt = all_data[i:i+1,t_his:,:]
        gt_multi = traj_gt_arr[i]
        for algo in algos:
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt, gt_multi) / num_seeds
                stats_meter[stats][algo].update(val)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)

    with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)




if __name__ == '__main__':
    all_algos = ['vae','dlow']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None) # specify the model to evaluate
    parser.add_argument('--mode', default='vis') # for visualization or quantitative results?
    parser.add_argument('--testdata', default='ACCAD') # which dataset to evaluate? choose only one
    parser.add_argument('--gpu_index', type=int, default=-1)

    ### these are better not to be touched ###
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter', type=int, default=500)
    ### these are better not to be touched ###
    args = parser.parse_args()

    if 'mojo' in args.cfg:
        from models.models_mojo import *
    elif 'vanilla' in args.cfg:
        from models.models_vanilla import *

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    body_repr = cfg.body_repr
    subsets = cfg.dataset
    n_freq = cfg.dlow_specs.get('n_freq', None)


    """data"""
    testing_data = [args.testdata]
    if len(testing_data)>1:
        raise NameError('performing testing per dataset please.')
    batch_gen = BatchGeneratorAMASSCanonicalized(amass_data_path=cfg.dataset_path,
                                                 amass_subset_name=testing_data,
                                                 sample_rate=8,
                                                 body_repr=body_repr)
    batch_gen.get_rec_list(shuffle_seed=3)
    all_data = batch_gen.get_all_data().detach().cpu().permute(1,0,2).numpy()#[b,t,d]
    traj_gt_arr = get_multimodal_gt()

    """models"""
    model_generator = {
        'vae': get_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, batch_gen.get_feature_dim())
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, 'iter')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = torch.load(model_path, map_location='cuda:0')
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    if args.mode == 'vis':
        visualize(n_seqs=60, n_gens=15)
    elif args.mode == 'stats':
        compute_stats()
