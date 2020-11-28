import os
import sys
import numpy as np
import open3d as o3d
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
import torch
import smplx
import cv2
import pickle
import pdb
import argparse
import json


def get_body_model(type, gender, batch_size,device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = '/home/yzhang/body_models/VPoser'
    body_model = smplx.create(body_model_path, model_type=type,
                                    gender=gender, ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=batch_size
                                    )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

def evaluation(gender, betas, transf_rotmat, transf_transl,
                data, datatype='gt'):

    ## parse smplx parameters
    bm = get_body_model('smplx', gender, 60, device='cpu')
    bparam = {}
    bparam['transl'] = data[:,:3]
    bparam['global_orient'] = data[:,3:6]
    bparam['betas'] = np.tile(betas[None,...], (60,1))
    bparam['body_pose'] = data[:,6:69]
    bparam['left_hand_pose'] = data[:,69:81]
    bparam['right_hand_pose'] = data[:,81:]

    ## from amass coord to world coord
    global_ori_a = R.from_rotvec(bparam['global_orient']).as_dcm() # to [t,3,3] rotation mat
    global_ori_w = np.einsum('ij,tjk->tik', transf_rotmat, global_ori_a)
    bparam['global_orient'] = R.from_dcm(global_ori_w).as_rotvec()
    bparam['transl'] = np.einsum('ij,tj->ti', transf_rotmat, bparam['transl']) + transf_transl

    ## obtain body mesh sequences
    for key in bparam:
        bparam[key] = torch.FloatTensor(bparam[key])
    verts_seq = bm(return_verts=True, **bparam).vertices.detach().cpu().numpy() #[t,verts, 3]

    ## get vertices at the feetbottom
    verts_feet = verts_seq[:,feetmarkeridx,:]
    verts_feet_horizon_vel = np.linalg.norm(verts_feet[1:, :, :-1]-verts_feet[:-1,:, :-1], axis=-1)[14:]
    verts_feet_height = verts_seq[15:,feetmarkeridx,-1]
    thresh_height = 5e-2
    thresh_vel = 5e-3
    skating = (verts_feet_horizon_vel>thresh_vel)*(np.abs(verts_feet_height)<thresh_height)
    skating = np.sum(np.logical_and(skating[:,0], skating[:,1])) /45

    return skating


if __name__=='__main__':
    proj_path = os.getcwd()
    with open('/home/yzhang/body_models/Mosh_related/CMU.json') as f:
        markerdict = json.load(f)['markersets'][0]['indices']
    feetmarkeridx = [markerdict['LHEE'], markerdict['RHEE']]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ACCAD')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--algo', default='dlow', help='dlow or gt')
    args = parser.parse_args()

    results_file_name = proj_path+'/results/{}/results/seq_gen_seed1_optim/{}/results_marker_41.pkl'.format(args.cfg,args.dataset)
    print('-- processing: '+results_file_name)

    with open(results_file_name, 'rb') as f:
        data = pickle.load(f)
    algos = [args.algo]
    skatings= []
    pene_vol= []
    for algo in algos:
        dd = data[algo+'_smplx_params']
        n_seq=dd.shape[0]
        n_gen=dd.shape[1]
        for seq in range(n_seq):
            gender = data['gender'][seq]
            betas = data['betas'][seq]
            transf_rotmat = data['transf_rotmat'][seq]
            transf_transl = data['transf_transl'][seq]
            for gen in range(n_gen):
                sks, pla=evaluation(gender, betas, transf_rotmat, transf_transl,
                            dd[seq, gen], datatype=algo)
                skatings.append(sks)
        if args.mode == 'stats':
            print('[results] skating={:f}, coll_vol={:f}'.format(np.mean(skatings),
                                                                     np.mean(pene_vol)))
