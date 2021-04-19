import os
import sys
import pickle
import numpy as np
import torch
import smplx
import json
from tqdm import tqdm
import pdb
import copy

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchgeometry as tgm
sys.path.append(os.getcwd())
from human_body_prior.tools.model_loader import load_vposer



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
        self.bm = None
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.to(self.device)
        self.vposer.eval()

        ## setup optim variables
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
        elif fittingconfig['num_markers'] == 67:
            with open('/home/yzhang/body_models/Mosh_related/SSM2.json') as f:
                self.marker = list(json.load(f)['markersets'][0]['indices'].values())



    def set_motion_model(self, model):
        self.motion_model = copy.deepcopy(model)
        self.motion_model.eval()


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

        return batch_jts.reshape(batch_jts.shape[0], -1) #[b, 22*3]


    def calc_loss(self, betas, verts_gt, stage):
        '''
        verts_gt: [t, num_markers, 3]
        '''
        body_param = {}
        body_param['transl'] = self.transl_rec
        body_param['global_orient'] = RotConverter.rotmat2aa(RotConverter.cont2rotmat(self.glo_rot_rec))
        body_param['betas'] = torch.tensor(betas, dtype=torch.float32, requires_grad=False).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)
        body_param['body_pose'] = self.vposer.decode(self.vpose_rec,
                                           output_type='aa').view(self.batch_size, -1)
        body_param['left_hand_pose'] = self.hand_pose[:,:12]
        body_param['right_hand_pose'] = self.hand_pose[:,12:]

        verts_full = self.bm(return_verts=True, **body_param).vertices
        verts = verts_full[:,self.marker,:]

        loss = (torch.mean( (verts-verts_gt.detach())**2)
                + (stage>1)*0.01*torch.mean(self.hand_pose**2)
                + (stage>0)*0.0005*torch.mean(self.vpose_rec**2))

        return loss, verts


    def fitting_subloop(self, gender, betas,
                        prev_transl, prev_glorot, prev_pose, prev_handpose,
                        curr_markers):
        verts_gt = curr_markers.view(-1,self.num_markers,3)
        self.transl_rec.data = prev_transl.detach().clone()
        self.glo_rot_rec.data = prev_glorot.detach().clone()
        self.vpose_rec.data = prev_pose.detach().clone()
        self.hand_pose.data = prev_handpose.detach().clone()

        for ss, optimizer in enumerate(self.optimizers):
            for ii in range(self.num_iter[ss]):
                optimizer.zero_grad()
                loss, verts= self.calc_loss(betas, verts_gt, ss)
                loss.backward(retain_graph=False)
                optimizer.step()
                if self.verbose:
                    print('[INFO][fitting][stage{:d}] iter={:d}, loss={:f}'.format(ss,
                                            ii, loss.item()) )

        ## update prev variables
        transl_out = self.transl_rec.detach().clone()
        glorot_out = self.glo_rot_rec.detach().clone()
        pose_out = self.vpose_rec.detach().clone()
        hand_pose_out = self.hand_pose.detach().clone()
        verts_out = verts.detach().clone().view(-1, self.num_markers*3)
        return verts_out, transl_out, glorot_out, pose_out, hand_pose_out


    def decode_with_fitting(self, x, z,
                            gender, prev_betas,
                            prev_transl, prev_glorot,
                            prev_pose, prev_handpose):

        ## set body model
        if gender == 'male':
            self.bm = self.bm_male
        elif gender == 'female':
            self.bm = self.bm_female

        h_x = self.motion_model.encode_x(x.detach())
        freqbasis = getattr(self.motion_model, 'freqbasis',[])
        if 'dct' in freqbasis:
            z = torch.einsum('wt,wbd->tbd',self.motion_model.dct_mat, z.detach())
        #intial gru states
        h_d = self.motion_model.drnn_mlp(h_x)
        self.motion_model.d_rnn.initialize(batch_size=z.shape[1], hx=h_d)
        y = []
        est_transl = []
        est_glorot = []
        est_pose = []
        est_handpose=[]
        est_transl.append(prev_transl)
        est_glorot.append(prev_glorot)
        est_pose.append(prev_pose)
        est_handpose.append(prev_handpose)
        for i in range(self.motion_model.horizon):
            y_p = x[-1] if i == 0 else verts_out
            ## network inference
            if len(freqbasis)==0:
                rnn_in = torch.cat([h_x, z, y_p.detach()], dim=1)
            else:
                rnn_in = torch.cat([h_x, z[i], y_p.detach()], dim=1)
            h = self.motion_model.d_rnn(rnn_in)
            h = self.motion_model.d_mlp(h)
            y_i = self.motion_model.d_out(h)
            y_i += y_p

            ## fitting
            [verts_out,
             prev_transl_,
             prev_glorot_,
             prev_pose_,
             prev_handpose_] = self.fitting_subloop(gender, prev_betas,
                                    est_transl[-1].detach(), est_glorot[-1].detach(),
                                    est_pose[-1].detach(), est_handpose[-1].detach(),
                                    y_i.detach())
            ## update results list
            y.append(verts_out)
            est_transl.append(prev_transl_)
            est_glorot.append(prev_glorot_)
            est_pose.append(prev_pose_)
            est_handpose.append(prev_handpose_)


        y = torch.stack(y)
        est_transl = torch.stack(est_transl[1:])
        est_glorot_cont = torch.stack(est_glorot[1:])
        est_glorot_aa = RotConverter.rotmat2aa(RotConverter.cont2rotmat(est_glorot_cont)).view(est_transl.shape[0],-1,3)
        est_pose_vp = torch.stack(est_pose[1:])
        est_pose_aa = self.vposer.decode(est_pose_vp,output_type='aa').view(est_transl.shape[0],-1,21*3)
        est_handpose = torch.stack(est_handpose[1:])
        y_smplxparams = torch.cat([est_transl, est_glorot_aa, est_pose_aa,est_handpose],dim=-1)
        return y, y_smplxparams

