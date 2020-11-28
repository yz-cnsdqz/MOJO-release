import numpy as np
import argparse
import os
import sys
import pickle
import csv
from tqdm import trange

sys.path.append(os.getcwd())
from utils import *
from experiments.utils.config import Config
from experiments.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from experiments.utils.eval_metrics import *
from models.fittingop import FittingOP


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



def visualize(data=None, n_seq=60, n_gen=50):
    num_seeds = args.num_seeds
    gen_results = {}
    gen_results['jts_pred'] = []
    gen_results['betas'] = []
    gen_results['gender'] = []
    gen_results['transf_rotmat'] = []
    gen_results['transf_transl'] = []
    for idx in range(n_seq):
        print('-- process sequence {:d}'.format(idx))
        data0 = batch_gen.next_sequence()
        traj0 = data0['body_feature']
        gen_results['betas'].append(data0['betas'])
        gen_results['gender'].append(str(data0['gender']))
        gen_results['transf_rotmat'].append(data0['transf_rotmat'])
        gen_results['transf_transl'].append(data0['transf_transl'])
        traj = traj0[None, :t_his]
        gt = traj0[None, t_his:]
        gt_multi = traj_gt_arr[idx]
        if data is None:
            pred = get_prediction_stats(traj, 'dlow', sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=True)
            gen_results['jts_pred'].append(pred[0])
        else:
            raise NameError('it is not necessary to save the data again. Check your previous results.'
        )
    ### save to file for later visualization
    gen_results['jts_pred'] = np.stack(gen_results['jts_pred'])
    outfilename = '{}/seq_gen_seed{}_optim/{}/'.format(cfg.result_dir, args.num_seeds, testing_data[0])
    if not os.path.exists(outfilename):
        os.makedirs(outfilename)
    outfilename += 'results_{}.pkl'.format(body_repr)
    with open(outfilename, 'wb') as f:
        pickle.dump(gen_results, f)


def compute_stats(data=None, n_seq=60, n_gen=50):
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde,
                  'BDFORM': compute_bone_deform}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}
    num_seeds = args.num_seeds

    for idx in range(n_seq):
        print('-- process sequence {:d}'.format(idx))
        data0 = batch_gen.next_sequence()
        traj0 = data0['body_feature']
        gen_results['betas'].append(data0['betas'])
        gen_results['gender'].append(str(data0['gender']))
        gen_results['transf_rotmat'].append(data0['transf_rotmat'])
        gen_results['transf_transl'].append(data0['transf_transl'])

        traj = traj0[None, :t_his]
        gt = traj0[None, t_his:]
        gt_multi = traj_gt_arr[idx]
        if data is None:
            pred = get_prediction(traj, 'dlow', sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
        else:
            data_param = data['dlow_smplx_params'][idx][:,t_his:] #[gen, t, d]
            data_betas = data['betas'][idx]
            data_genders = data['gender'][idx]
            pred = []
            for t in range(data_param.shape[1]):
                pred.append(fittingop.get_jts_from_smplx(data_betas,data_genders, data_param[:,t]))
            pred = np.stack(pred, axis=1)[None,...] #[1,gen,t,d]
        pred = pred[:,:n_gen]

        for stats in stats_names:
            val = 0
            for pred_i in pred:
                val += stats_func[stats](pred_i, gt, gt_multi) / num_seeds
            stats_meter[stats]['dlow'].update(val)

    ### log or show scores
    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)





if __name__ == '__main__':
    all_algos = ['vae','dlow']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='vis')
    parser.add_argument('--testdata', default='ACCAD')
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--bmresults', default=None) # the body mesh results obtained before

    ## better not to touch the following setting.
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--iter', type=int, default=500)
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
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(True)
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
                                                 body_repr='joints')
    batch_gen.get_rec_list(shuffle_seed=3)
    all_data = batch_gen.get_all_data().detach().cpu().permute(1,0,2).numpy() #[b,t,d]
    traj_gt_arr = get_multimodal_gt(all_data, t_his, args.multimodal_threshold)

    """models"""
    model_generator = {
        'vae': get_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, 66)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, 'iter')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = torch.load(model_path, map_location='cuda:0')
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()


    """ fitting configures
        here we only use it to get joints from body meshes
    """
    fittingconfig={ 'init_lr_h': 0.008,
                    'num_iter': [10,30,20],
                    'batch_size': cfg.nk,
                    'num_markers': 41,
                    'device': device,
                    'verbose': False
                }
    fittingop = FittingOP(fittingconfig)

    if args.mode == 'vis':
        visualize(n_seqs=60, n_gens=50)
    elif args.mode == 'stats':
        results_file_name = args.bmresults
        if results_file_name is None:
            compute_stats(data=None, n_seq=60, n_gen=50)
        else:
            with open(results_file_name, 'rb') as f:
                data = pickle.load(f)
            compute_stats(data=data, n_seq=60, n_gen=50)
