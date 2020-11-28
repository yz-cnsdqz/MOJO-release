import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fftn


def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, ord=2, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, ord=2, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_bone_deform(gen, gt, gt_multi):
    '''
    gen, gt - [nsamp, time, dim]
    '''
    jts = gen.reshape([gen.shape[0], gen.shape[1],22,3]) #[gen, t, 22, 3]
    l_LFA = np.linalg.norm(jts[:,:,18]-jts[:,:,20], axis=-1).std(axis=-1).mean()
    l_LUA = np.linalg.norm(jts[:,:,18]-jts[:,:,16], axis=-1).std(axis=-1).mean()
    l_RUA = np.linalg.norm(jts[:,:,19]-jts[:,:,17], axis=-1).std(axis=-1).mean()
    l_RFA = np.linalg.norm(jts[:,:,19]-jts[:,:,21], axis=-1).std(axis=-1).mean()
    l_LTH = np.linalg.norm(jts[:,:,1]-jts[:,:,4], axis=-1).std(axis=-1).mean()
    l_LCA = np.linalg.norm(jts[:,:,7]-jts[:,:,4], axis=-1).std(axis=-1).mean()
    l_RTH = np.linalg.norm(jts[:,:,2]-jts[:,:,5], axis=-1).std(axis=-1).mean()
    l_RCA = np.linalg.norm(jts[:,:,5]-jts[:,:,8], axis=-1).std(axis=-1).mean()
    deform = l_LFA+l_LUA+l_RUA+l_RFA+l_LTH+l_LCA+l_RTH+l_RCA
    return deform

def compute_ps_entropy(gen, gt, gt_multi):
    '''
    gen, gt - [nsamp, time, dim]
    '''
    ### ps entropy
    ps_gen = np.abs(fftn(gen, axes=1))**2 + 1e-6
    ps_gen = ps_gen / np.sum(ps_gen, axis=1, keepdims=True)
    ps_entropy_gen =  np.mean(-ps_gen*np.log(ps_gen),axis=-1)

    ps_gt = np.abs(fftn(gt, axes=1))**2 + 1e-6
    ps_gt = ps_gt / np.sum(ps_gt, axis=1, keepdims=True)
    ps_entropy_gt =  np.mean(-ps_gt*np.log(ps_gt), axis=-1)

    return np.mean(ps_entropy_gen-ps_entropy_gt)


def get_multimodal_gt(all_data, t_his, thresh):
    all_start_pose = all_data[:,t_his - 1,:]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < thresh)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
    return traj_gt_arr
