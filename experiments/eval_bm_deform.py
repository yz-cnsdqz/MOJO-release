import os
import sys
import numpy as np
import open3d as o3d
sys.path.append(os.getcwd())
import torch
import pickle
import pdb
import argparse
import json
from scipy.spatial.distance import pdist


def calc_pdist_variations(seq):
    nframes = seq.shape[0]
    seqpdist = np.stack([pdist(seq[i]) for i in range(nframes) ])
    return np.sum(np.std(seqpdist, axis=0))


def calc_metrics(data):
    data = data[15:]
    deform_hd = calc_pdist_variations(data[:,head_marker_idx,:])
    deform_ut = calc_pdist_variations(data[:,uppertorso_marker_idx,:])
    deform_lt = calc_pdist_variations(data[:,lowertorso_marker_idx,:])

    return deform_hd, deform_ut, deform_lt



if __name__=='__main__':


    proj_path = os.getcwd()
    with open('/home/yzhang/body_models/Mosh_related/CMU.json') as f:
                markerdict = json.load(f)['markersets'][0]['indices']

    markeridx = list(markerdict.values())
    head_marker_idx = [markeridx.index(markerdict['RFHD']),
                       markeridx.index(markerdict['LFHD']),
                       markeridx.index(markerdict['RBHD']),
                       markeridx.index(markerdict['LBHD'])]
    uppertorso_marker_idx = [markeridx.index(markerdict['RSHO']),
                             markeridx.index(markerdict['LSHO']),
                             markeridx.index(markerdict['CLAV']),
                             markeridx.index(markerdict['C7'])]
    lowertorso_marker_idx = [markeridx.index(markerdict['RFWT']),
                             markeridx.index(markerdict['LFWT']),
                             markeridx.index(markerdict['LBWT']),
                             markeridx.index(markerdict['RBWT'])]


    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='pred_optim1')
    parser.add_argument('--dataset', default='ACCAD')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--algo', default='dlow', help='dlow or gt')

    args = parser.parse_args()



    exp = args.cfg
    if args.mode == 'pred_optim':
        results_file_name = proj_path+'/results/{}/results/seq_gen_seed1_optim/{}/results_marker_41.pkl'.format(exp,args.dataset)
    elif args.mode == 'pred':
        results_file_name = proj_path+'/results/{}/results/seq_gen_seed1/{}/results_marker_41.pkl'.format(exp,args.dataset)

    print('-- processing: '+results_file_name)

    with open(results_file_name, 'rb') as f:
        data = pickle.load(f)
    algo = args.algo
    deform_head= []
    deform_ut= []
    deform_lt= []
    dd = data[algo]
    n_seq=dd.shape[0]
    n_gen=dd.shape[1]
    for seq in range(n_seq):
        for gen in range(n_gen):
            dhead, dut, dlt = calc_metrics(dd[seq, gen])
            deform_head.append(dhead)
            deform_ut.append(dut)
            deform_lt.append(dlt)
    df_hd = np.mean(deform_head)
    df_ut = np.mean(deform_ut)
    df_lt = np.mean(deform_lt)

    print('[results] head deformation={:f}, upper torso deformation={:f}, lower torso deformation={:f}'\
                        .format(df_hd,df_ut,df_lt))
