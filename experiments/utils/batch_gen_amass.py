from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
import glob
import os, sys
from scipy.spatial.transform import Rotation as R
import smplx
import pdb


class BatchGeneratorAMASSCanonicalized(object):
    def __init__(self,
                amass_data_path,
                amass_subset_name=None,
                sample_rate=2, 
                body_repr='smpl_params' #['smpl_params', 'marker_41', 'marker_67', 'joints']
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.amass_data_path = amass_data_path
        self.amass_subset_name = amass_subset_name
        self.sample_rate = sample_rate
        self.data_list = []
        self.body_repr = body_repr

    def reset(self):
        self.index_rec = 0
        random.shuffle(self.data_list)
        if self.body_repr in ['joints', 'marker_41', 'marker_67']:
            np.random.shuffle(self.data_all)

    def has_next_rec(self):
        if self.index_rec < len(self.data_list):
            return True
        return False

    def get_rec_list(self, shuffle_seed=None):

        if self.amass_subset_name is not None:
            ## read the sequence in the subsets
            self.rec_list = []
            for subset in self.amass_subset_name:
                self.rec_list += glob.glob(os.path.join(self.amass_data_path,
                                                       subset,
                                                       '*.npz'  ))
        else:
            ## read all amass sequences
            self.rec_list = glob.glob(os.path.join(self.amass_data_path,
                                                    '*/*.npz'))

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)
        else:
            random.shuffle(self.rec_list) # shuffle recordings, not frames in a recording.

        print('[INFO] read all data to RAM...')
        for rec in self.rec_list:
            pose = np.load(rec)['poses'][::self.sample_rate,:66] # 156d = 66d+hand
            transl = np.load(rec)['trans'][::self.sample_rate]
            if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                continue
            body_marker_41 = np.load(rec)['marker_cmu_41'][::self.sample_rate].reshape([-1,41*3])
            body_marker_67 = np.load(rec)['marker_ssm2_67'][::self.sample_rate].reshape([-1,67*3])
            joints = np.load(rec)['joints'][::self.sample_rate].reshape([-1,22*3])
            
            body_feature = {}
            if self.body_repr == 'smpl_params':
                body_feature['transl'] = transl
                body_feature['pose'] = pose
            elif self.body_repr == 'joints':
                body_feature = joints
            elif self.body_repr == 'marker_41':
                body_feature = body_marker_41
            elif self.body_repr == 'marker_67':
                body_feature = body_marker_67
            else:
                raise NameError('[ERROR] not valid body representation. Terminate')
            self.data_list.append(body_feature)

        if self.body_repr in ['joints', 'marker_41', 'marker_67']:
            self.data_all = np.stack(self.data_list,axis=0) #[b,t,d]


    def next_batch_smplx_params(self, batch_size=64):
        batch_pose_ = []
        batch_transl_ = []
        ii = 0
        while ii < batch_size:
            if not self.has_next_rec():
                break
            data = self.data_list[self.index_rec]
            batch_tensor_pose = torch.FloatTensor(data['pose']).unsqueeze(0)
            batch_tensor_transl = torch.FloatTensor(data['transl']).unsqueeze(0) #[b,t,d]
            batch_pose_.append(batch_tensor_pose)
            batch_transl_.append(batch_tensor_transl)
            ii = ii+1
            self.index_rec+=1
        batch_pose = torch.cat(batch_pose_,dim=0).permute(1,0,2) #[t,b,d]
        batch_transl = torch.cat(batch_transl_,dim=0).permute(1,0,2) #[t,b,d]
        return [batch_pose, batch_transl]


    def next_batch_kps(self, batch_size=64):
        batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size]
        self.index_rec+=batch_size
        batch_data = torch.FloatTensor(batch_data_).permute(1,0,2) #[t,b,d]

        return batch_data


    def next_batch(self, batch_size=64):
        '''
        the key funtion to generate batch
        '''
        if self.body_repr == 'smpl_params':
            batch = self.next_batch_smplx_params(batch_size)
        else:
            batch = self.next_batch_kps(batch_size)
        return batch


    def next_sequence(self):
        '''
        - this function is only for produce files for visualization or testing in some cases
        - compared to next_batch with batch_size=1, this function also outputs metainfo, like gender, body shape, etc.
        '''
        rec = self.rec_list[self.index_rec]
        pose = np.load(rec)['poses'][::self.sample_rate,:66] # 156d = 66d+hand
        transl = np.load(rec)['trans'][::self.sample_rate]
        if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
            return None
        body_shape = np.load(rec)['betas'][:10]
        body_marker_41 = np.load(rec)['marker_cmu_41'][::self.sample_rate]
        body_marker_67 = np.load(rec)['marker_ssm2_67'][::self.sample_rate]
        joints = np.load(rec)['joints'][::self.sample_rate]

        body_feature = {}
        if self.body_repr == 'smpl_params':
            body_feature['transl'] = transl
            body_feature['pose'] = pose
        elif self.body_repr == 'joints':
            body_feature = joints.reshape([-1,22*3])
        elif self.body_repr == 'marker_41':
            body_feature = body_marker_41.reshape([-1,41*3])
        elif self.body_repr == 'marker_67':
            body_feature = body_marker_67.reshape([-1,67*3])
        else:
            raise NameError('[ERROR] not valid body representation. Terminate')

        ## pack output data
        output = {}
        output['betas'] = body_shape
        output['gender'] = np.load(rec)['gender']
        output['transl'] = transl
        output['glorot'] = pose[:,:3]
        output['poses'] = pose[:,3:]
        output['body_feature'] = body_feature
        output['transf_rotmat'] = np.load(rec)['transf_rotmat']
        output['transf_transl'] = np.load(rec)['transf_transl']

        self.index_rec += 1
        return output

    def get_feature_dim(self):
        if self.body_repr == 'smpl_params':
            raise NameError('return list. No dim')
        elif self.body_repr == 'marker_41':
            return 41*3
        elif self.body_repr == 'marker_67':
            return 67*3
        elif self.body_repr == 'joints':
            return 22*3
        else:
            raise NameError('not implemented')


    def get_all_data(self):
        return torch.FloatTensor(self.data_all).permute(1,0,2) #[t,b,d]


### an example of how to use it
if __name__=='__main__':
    batch_gen = BatchGeneratorAMASSCanonicalized(amass_data_path='/home/yzhang/Videos/AMASS-Canonicalized/data/',
                                                 amass_subset_name=['ACCAD'],
                                                 sample_rate=8,
                                                 body_repr='marker_41')
    batch_gen.get_rec_list()
    data = batch_gen.next_batch(batch_size=64)
    print(data[0])
    data = batch_gen.next_batch(batch_size=64)
    print(data[0])













