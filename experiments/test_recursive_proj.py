import numpy as np
import argparse
import os
import sys
import pickle
import csv
from scipy.spatial.distance import pdist
from tqdm import trange

sys.path.append(os.getcwd())
from human_body_prior.tools.model_loader import load_vposer
from utils import *
from experiments.utils.config import Config
from experiments.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from experiments.utils.visualization import render_animation
from scipy.spatial.distance import pdist, squareform
from models.fittingop import RotConverter, FittingOP


def get_prediction(smplxparams, traj_np, algo, sample_num, num_seeds=1, concat_hist=True,
                    gender=None, prev_betas=None,
                    prev_transl=None, prev_glorot_cont=None,
                    prev_pose_vp=None, prev_handpose=None):
    smplxparams = smplxparams[None,...].transpose([1,0,2])#[1,t,d]
    smplxparams_his = smplxparams[:t_his] #[t,1,d]
    traj = traj_np.permute(1, 0, 2).contiguous().to(device)
    X = traj[:t_his] # original setting
    if algo == 'dlow':
        X = X.repeat((1, num_seeds, 1))
        Z_g = models[algo].sample(X)
        if 'mojo' in args.cfg:
            Z_highfreq = torch.randn((t_pred-n_freq, Z_g.shape[1], Z_g.shape[2]),
                                    device=Z_g.device)
            Z_g = torch.cat([Z_g, Z_highfreq],dim=0) #[freq, b*nk, d]
        X = X.repeat_interleave(sample_num, dim=1)
        Y, smplxparams_pred = fittingop.decode_with_fitting(X, Z_g,
                                            gender, prev_betas,
                                            prev_transl, prev_glorot_cont,
                                            prev_pose_vp, prev_handpose)
    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
        smplxparams_pred = smplxparams_pred.detach().cpu().numpy()
        smplxparams_his = np.tile(smplxparams_his, [1,cfg.nk, 1])
        smplxparams = np.concatenate([smplxparams_his, smplxparams_pred])
    Y = Y.permute(1, 0, 2).contiguous().detach().cpu().numpy() #[nk, time, dim]
    smplxparams = smplxparams.transpose([1,0,2])

    return Y, smplxparams



def pred_with_proj(n_seqs, n_gens):
    '''
    n_seqs: how many sequences to generate
    n_gens: for each input sequence, how many different sequences to predict
    '''
    gen_results = {}
    gen_results['gt_marker'] = []
    gen_results['gt_smplx_params'] = []
    gen_results['betas'] = []
    gen_results['gender'] = []
    gen_results['transf_rotmat'] = []
    gen_results['transf_transl'] = []
    for algo in vis_algos:
        gen_results[algo] = []
        gen_results[algo+'_smplx_params'] = []
    for idx in range(n_seqs):
        print('-- process sequence {:d}'.format(idx))
        data = batch_gen.next_sequence()
        ## prepare variables to save
        motion_np = data['body_feature']
        motion = torch.FloatTensor(motion_np).unsqueeze(0) #[b,t,d]
        gen_results['gt_marker'].append(motion_np.reshape((1,motion_np.shape[0],-1,3)))
        smplx_transl = data['transl']
        smplx_glorot = data['glorot']
        smplx_poses = data['poses']
        smplx_handposes = np.zeros([smplx_transl.shape[0], 24])
        motion_smplxparam = np.concatenate([smplx_transl,smplx_glorot,
                                                smplx_poses,smplx_handposes],axis=-1) #[t,d]
        gen_results['gt_smplx_params'].append(motion_smplxparam[None,...])
        gen_results['betas'].append(data['betas'])
        gen_results['gender'].append(str(data['gender']))
        gen_results['transf_rotmat'].append(data['transf_rotmat'])
        gen_results['transf_transl'].append(data['transf_transl'])

        # initialize the states for optimization
        prev_transl = torch.FloatTensor(data['transl'][t_his-1:t_his]).to(device).repeat(cfg.nk, 1) #[nk,3]
        prev_glorot_aa = torch.FloatTensor(data['glorot'][t_his-1:t_his]).to(device) #[1,3]
        prev_glorot_cont = RotConverter.aa2cont(prev_glorot_aa.unsqueeze(0))[0,0].repeat(cfg.nk, 1) #[nk,6]
        prev_pose_aa = torch.FloatTensor(data['poses'][t_his-1:t_his]).to(device)
        prev_pose_vp = fittingop.vposer.encode(prev_pose_aa).mean.repeat(cfg.nk, 1)
        prev_handpose = torch.FloatTensor(size=(cfg.nk,24)).zero_().to(device)
        prev_betas = data['betas']
        gender = str(data['gender'])

        # only for dlow, [1:] is necessary
        for algo in vis_algos[1:]:
            pred, pred_params = get_prediction(motion_smplxparam, motion, algo, cfg.nk, 1,True,
                                gender, prev_betas,
                                prev_transl, prev_glorot_cont,
                                prev_pose_vp, prev_handpose)
            pred = np.reshape(pred, (pred.shape[0], pred.shape[1],-1,3))
            pred = pred[:n_gens]
            pred_params = pred_params[:n_gens]
            gen_results[algo].append(pred)
            gen_results[algo+'_smplx_params'].append(pred_params)
        idx+=1
    gen_results['gt_marker'] = np.stack(gen_results['gt_marker'])
    gen_results['gt_smplx_params'] = np.stack(gen_results['gt_smplx_params'])

    for algo in vis_algos[1:]: #load vae but dont use it.
        gen_results[algo] = np.stack(gen_results[algo]) #[#seq, #genseq_per_pastmotion, t, #joints, 3]
        gen_results[algo+'_smplx_params'] = np.stack(gen_results[algo+'_smplx_params'])
    ### save to file
    outfilename = '{}/seq_gen_seed{}_optim/{}/'.format(cfg.result_dir, args.num_seeds, testing_data[0])
    if not os.path.exists(outfilename):
        os.makedirs(outfilename)
    outfilename += 'results_{}.pkl'.format(body_repr)
    with open(outfilename, 'wb') as f:
        pickle.dump(gen_results, f)



def get_multimodal_gt():
    all_start_pose = all_data[:,t_his - 1,:]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
    return traj_gt_arr


if __name__ == '__main__':
    all_algos = ['vae','dlow']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--testdata', default='ACCAD')
    parser.add_argument('--gpu_index', type=int, default=0)

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
                                                 body_repr=body_repr)
    batch_gen.get_rec_list(shuffle_seed=3)
    all_data = batch_gen.get_all_data().detach().cpu().permute(1,0,2).numpy() #[b,t,d]
    traj_gt_arr = get_multimodal_gt()
    n_markers = batch_gen.get_feature_dim()//3


    """models"""
    model_generator = {
        'vae': get_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, n_markers*3)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, 'iter')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = torch.load(model_path, map_location='cuda:0')
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    """fitting configures"""
    fittingconfig={ 'init_lr_h': 0.008,
                    'num_iter': [10,30,20],
                    'batch_size': cfg.nk,
                    'num_markers': n_markers, 
                    'device': device,
                    'verbose': False
                }
    fittingop = FittingOP(fittingconfig)
    fittingop.set_motion_model(models['vae'])
    pred_with_proj(n_seqs=60, n_gens=50)
