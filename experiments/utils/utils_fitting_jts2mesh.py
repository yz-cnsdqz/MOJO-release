import os
import sys
import pickle
import numpy as np
import torch
import smplx
import json
from tqdm import tqdm
import pdb

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchgeometry as tgm
sys.path.append(os.getcwd())
from human_body_prior.tools.model_loader import load_vposer
import argparse


class RotConverter(nn.Module):
    '''
    this class is from smplx/vposer
    '''
    def __init__(self):
        super(RotConverter, self).__init__()

    def forward(self,module_input):
        pass


    @staticmethod
    def cont2rotmat(module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def aa2cont(module_input):
        '''
        :param NxTxnum_jointsx3
        :return: pose_matrot: NxTxnum_jointsx6
        '''
        batch_size = module_input.shape[0]
        n_frames = module_input.shape[1]
        pose_body_6d = tgm.angle_axis_to_rotation_matrix(module_input.reshape(-1, 3))[:, :3, :2].contiguous().view(batch_size, n_frames, -1, 6)

        return pose_body_6d


    @staticmethod
    def rotmat2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()

        return pose


    @staticmethod
    def aa2rotmat(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.shape[0]
        n_frames = pose.shape[1]
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, n_frames, -1, 9)

        return pose_body_matrot





class FittingOP:
    def __init__(self, fittingconfig):

        for key, val in fittingconfig.items():
            setattr(self, key, val)

        body_model_path = '/home/yzhang/body_models/VPoser'
        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
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
                                    batch_size=self.batch_size
                                    )
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
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
                                    batch_size=self.batch_size
                                    )
        self.bm_male.to(self.device)
        self.bm_female.to(self.device)
        self.bm_male.eval()
        self.bm_female.eval()
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.to(self.device)
        self.vposer.eval()

        self.transl_rec = Variable(torch.zeros(self.batch_size,3).to(self.device), requires_grad=True)
        self.glo_rot_rec = Variable(torch.FloatTensor([[-1,0,0,0,0,1]]).repeat(self.batch_size,1).to(self.device), requires_grad=True)
        self.vpose_rec = Variable(torch.zeros(self.batch_size,32).to(self.device), requires_grad=True)
        self.hand_pose = Variable(torch.randn(self.batch_size,24).to(self.device), requires_grad=True)
        self.optimizer_s1 = optim.Adam([self.transl_rec, self.glo_rot_rec], lr=self.init_lr_h)
        self.optimizer_s2 = optim.Adam([self.transl_rec, self.glo_rot_rec, self.vpose_rec], lr=self.init_lr_h)
        self.optimizer_s3 = optim.Adam([self.transl_rec, self.glo_rot_rec, self.vpose_rec, self.hand_pose],
                                        lr=self.init_lr_h)
        self.optimizers = [self.optimizer_s1, self.optimizer_s2, self.optimizer_s3]

        if fittingconfig['num_markers'] == 41:
            with open('/home/yzhang/body_models/Mosh_related/CMU.json') as f:
                self.marker = list(json.load(f)['markersets'][0]['indices'].values())

    def get_jts_from_smplx(self, betas, gender, data):
        bm = None
        if gender == 'male':
            bm = self.bm_male
        elif gender == 'female':
            bm = self.bm_female

        bparam = {}
        bparam['transl'] = data[:,:3]
        bparam['global_orient'] = data[:,3:6]
        bparam['betas'] = np.tile(betas[None,...], (data.shape[0],1))
        bparam['body_pose'] = data[:,6:69]
        bparam['left_hand_pose'] = data[:,69:81]
        bparam['right_hand_pose'] = data[:,81:]
        for key in bparam:
            bparam[key] = torch.FloatTensor(bparam[key]).to(self.device)
        batch_jts = bm(return_verts=True, **bparam).joints[:,:22,:].detach().cpu().numpy() #[b,jts, 3]

        return batch_jts



    def calc_loss(self, bm, betas, jts_gt, stage):
        '''
        jts_gt: [t, 22, 3]
        '''
        body_param = {}
        body_param['transl'] = self.transl_rec
        body_param['global_orient'] = RotConverter.rotmat2aa(RotConverter.cont2rotmat(self.glo_rot_rec))
        body_param['betas'] = torch.tensor(betas, dtype=torch.float32, requires_grad=False).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        body_param['body_pose'] = self.vposer.decode(self.vpose_rec,
                                           output_type='aa').view(self.batch_size, -1)
        body_param['left_hand_pose'] = self.hand_pose[:,:12]
        body_param['right_hand_pose'] = self.hand_pose[:,12:]

        jts = bm(return_verts=True, **body_param).joints[:,:22,:]
        loss = (torch.mean( (jts-jts_gt.detach())**2)
                + (stage>1)*0.01*torch.mean(self.hand_pose**2)
                + (stage>0)*0.0005 *torch.mean(self.vpose_rec**2))

        return loss



    def fitting(self, gender, betas, jts):

        jts_gt = torch.FloatTensor(jts).to(self.device).view(-1,22,3)
        self.transl_rec.data = torch.FloatTensor(self.batch_size,3).zero_().to(self.device)
        self.glo_rot_rec.data = torch.FloatTensor([[-1,0,0,0,0,1]]).repeat(self.batch_size,1).to(self.device)
        self.vpose_rec.data = torch.FloatTensor(self.batch_size,32).zero_().to(self.device)
        self.hand_pose.data = torch.cuda.FloatTensor(self.batch_size,24).zero_()

        bm = None
        if gender == 'male':
            bm = self.bm_male
        elif gender == 'female':
            bm = self.bm_female

        for ss, optimizer in enumerate(self.optimizers):
            for ii in range(self.num_iter[ss]):
                optimizer.zero_grad()
                loss = self.calc_loss(bm, betas, jts_gt, ss)
                loss.backward(retain_graph=True)
                optimizer.step()
                if self.verbose and ii%200==0 :
                    print('[INFO][fitting][stage{:d}] iter={:d}, loss={:f}'.format(ss,
                                            ii, loss.item()) )

        ## change data to a single vector [translation, glorot_aa, pose_aa]
        transl_out = self.transl_rec
        glorot_out = RotConverter.rotmat2aa(RotConverter.cont2rotmat(self.glo_rot_rec))
        pose_out = self.vposer.decode(self.vpose_rec,
                                           output_type='aa').view(self.batch_size, -1)
        hand_pose_out = self.hand_pose
        body_param_out = torch.cat([transl_out, glorot_out, pose_out, hand_pose_out],dim=-1)

        return body_param_out.clone().detach().cpu().numpy() #[t,d]




if __name__=='__main__':
    ## set mosh markers
    # ## read the corresponding smplx verts indices as markers.
    # with open('/home/yzhang/body_models/Mosh_related/CMU.json') as f:
    #         marker_cmu_41 = list(json.load(f)['markersets'][0]['indices'].values())
    fittingconfig={ 'init_lr_h': 0.008,
                    'num_iter': [50,150,100],
                    'batch_size': 60,
                    'num_markers': 41,
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    # 'device': torch.device('cpu'),
                    'verbose': False
                }
    fittingop = FittingOP(fittingconfig)

    ### get smplx parameters from markers
    proj_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None)
    args = parser.parse_args()

    results_file_name = args.filename
    print('-- processing: '+results_file_name)
    with open(results_file_name, 'rb') as f:
        data = pickle.load(f)

    betas = data['betas']
    gender = data['gender']
    dd = data['jts_pred']
    n_seq=dd.shape[0]
    n_gen=dd.shape[1]
    smplx_all = []
    for seq in tqdm(range(n_seq), position=1):
        smplx_all_=[]
        for gen in range(n_gen):
            jts = dd[seq, gen]
            params_rec = fittingop.fitting(gender[seq], betas[seq], jts)
            smplx_all_.append(params_rec)
        smplx_all_ = np.stack(smplx_all_)
        smplx_all.append(smplx_all_)
    smplx_all = np.stack(smplx_all)
    data['dlow_smplx_params'] = smplx_all

    with open(results_file_name, 'wb') as f:
        pickle.dump(data, f)