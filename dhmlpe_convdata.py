# Copyright (c) 2013, Li Sijin (lisijin7@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ibasic_convdata import *
##
import dhmlpe
import indicatormap
import dhmlpe_features
import dhmlpe_utils as dutils
###

class DHMLPEDataProviderError(Exception):
    pass
class CroppedDHMLPEJointDataProvider(CroppedImageDataProvider):
    """
    This data provider will provide
        [data, joints, joints_indicator_map]
        joints is the relateive joints
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        jt_filter_size = self.batch_meta['joint_indicator_map']['filter_size']
        jt_stride = self.batch_meta['joint_indicator_map']['stride']
        if ('joint_indmap_type' not in dp_params):
            ## default 
            self.joint_indmap = indicatormap.IndicatorMap(self.input_image_dim, \
                                                        jt_filter_size, \
                                                        jt_stride,
                                                        create_lookup_table=True)
        else:
            self.joint_indmap = indicatormap.IndMapDic[dp_params['joint_indmap_type']](\
                                                        self.input_image_dim, \
                                                        jt_filter_size, \
                                                        jt_stride,
                                                        create_lookup_table=True) 
        self.num_joints = self.batch_meta['num_joints']
        self.feature_name_3d = 'Relative_Y3d_mono_body'
        if 'max_depth' in self.batch_meta:
            self.max_depth = self.batch_meta['max_depth']
        else:
            self.max_depth = 1200 
        # self.max_depth = 1
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        ## Add joints here
        alldata += [np.require(self.data_dic['joints_3d'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')/self.max_depth]
        ## ADD joint indicator here
        alldata += [np.require(self.data_dic['joints_indicator_map'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        dic = CroppedImageDataProvider.get_batch(self,batch_num)
        ## ADD joint data here
        #print self.batch_meta['occ_body'].shape
        #print max(dic['cur_batch_indexes']), min(dic['cur_batch_indexes'])
        # Require square image
        dic['occ_body'] = np.concatenate(map(lambda x:self.batch_meta['occ_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)        
        dic['joints_2d'] = np.concatenate(map(lambda x:self.batch_meta['Y2d_bnd_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)
        dic['joints_3d'] = np.concatenate(map(lambda x:self.batch_meta[self.feature_name_3d][...,x].reshape((-1,1),order='F'),dic['cur_batch_indexes']),axis=1)
        ## generate joint indicator map
        pts = dic['joints_2d'].reshape((2,-1),order='F')
        #  subtract offset to get the cropped coordinates 
        offset = np.tile(np.concatenate([np.asarray(self.cur_offset_c).reshape((1,-1)), np.asarray(self.cur_offset_r).reshape((1,-1))],axis=0), [self.num_joints,1]).reshape((2,-1),order='F')
        pts = pts - offset # 0-index coordinates
        dic['joints_2d'] = pts.reshape((2*self.num_joints, -1),order='F')
        allmaps = self.joint_indmap.get_joints_indicatormap(pts.T)
        mdim = self.joint_indmap.mdim
        ndata = len(dic['cur_batch_indexes'])
        dic['joints_indicator_map'] = allmaps
        return dic
    def get_plottable_data(self, imgdata):
        ## it is different with base class's method
        # just to be consistent with previous testconvnet.py functions
        # Without transpose and scaling
        ndata = imgdata.shape[-1]
        dimX = imgdata.shape[0]
        res = imgdata.copy() +self.cropped_mean_image.reshape((dimX,1),order='F')
        imgdim = list(self.input_image_dim) + [ndata]
        return res.reshape(imgdim, order='F')
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * 3
        else:
            return iprod(self.joint_indmap.mdim) * self.num_joints


class CroppedDHMLPEJointDoubleDataProvider(CroppedDHMLPEJointDataProvider):
    """
    This data provider will provide
        [data, joints, joints_indicator_map,
         _data, _joints, _joints_indicator_map]
        joints is the relateive joints
        There are several ways to generate the duplicate
        In order to load efficiently, the second part of the data will
        come from the same batch.
        So that, images will only be loaded once.
        One can choose to simple reverse the order or randomly shuffle 
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.selection_type = 'shuffle'
    def get_next_batch(self):
        epoch, batch_num, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        selected_idx = range(ndata-1,-1,-1)
        if self.selection_type == 'shuffle':
            np.random.shuffle(selected_idx)
        l = []
        for elem in alldata:
            l += [np.require(elem[:,selected_idx].reshape((-1,ndata),order='F'),\
                             dtype=np.single, requirements='C')]
        alldata += l
        return epoch, batch_num, alldata
    def get_data_dims(self,idx=0):
        if idx < 3:
            return CroppedDHMLPEJointDataProvider.get_data_dims(self,idx)
        else:
            return CroppedDHMLPEJointDataProvider.get_data_dims(self,idx - 3)
        
class CroppedDHMLPEJointOccDataProvider(CroppedDHMLPEJointDataProvider):
    def get_batch(self, batch_num):
        dic = CroppedDHMLPEJointDataProvider.get_batch(self, batch_num)
        dic['joints_indicator_map'][..., dic['occ_body']] = False
        return dic


class CroppedDHMLPEDepthJointDataProvider(CroppedImageDataProvider):
    """
    This data provider will output:
      [data, joints, depth_joints_indicator_map]
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        jt_filter_size2d = self.batch_meta['joint_indicator_map']['filter_size']
        jt_stride2d = self.batch_meta['joint_indicator_map']['stride']
        self.max_depth = self.batch_meta['max_depth']
        ## 
        dimz = self.max_depth * 2
        ## Require those fields
        win_z = self.batch_meta['win_z']
        stride_z = self.batch_meta['stride_z']
        depthdim = [self.input_image_dim[1], self.input_image_dim[0], dimz]
        self.depth_joint_indmap = indicatormap.DepthIndicatorMap(depthdim, jt_filter_size2d, jt_stride2d, win_z, stride_z, create_lookup_table=True)
        self.num_joints = self.batch_meta['num_joints']
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        ## Add joints here
        alldata += [np.require(self.data_dic['joints_3d'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')/self.max_depth]
        ## ADD depth joint indicator here
        alldata += [np.require(self.data_dic['depth_joints_indicator_map'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        dic = CroppedImageDataProvider.get_batch(self,batch_num)
        dic['occ_body'] = np.concatenate(map(lambda x:self.batch_meta['occ_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)
        dic['joints_2d'] = np.concatenate(map(lambda x:self.batch_meta['Y2d_bnd_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)        
        dic['joints_3d'] = np.concatenate(map(lambda x:self.batch_meta['Relative_Y3d_mono_body'][...,x].reshape((-1,1)),dic['cur_batch_indexes']),axis=1)
        ## Get rectified 2d coordinates
        pts = dic['joints_2d'].reshape((2,-1),order='F')
        offset = np.tile(np.concatenate([np.asarray(self.cur_offset_c).reshape((1,-1)), np.asarray(self.cur_offset_r).reshape((1,-1))],axis=0), [self.num_joints,1]).reshape((2,-1),order='F')
        pts = pts - offset # 0-index coordinates
        dic['joints_2d'] = pts.reshape((self.num_joints*2, -1),order='F')
        depth_pts = np.concatenate([pts, dic['joints_3d'].reshape((3,-1))[2,:].reshape((1,-1),order='F')],axis=0).T        
        allmaps = self.depth_joint_indmap.get_joints_indicatormap(depth_pts)
        dic['depth_joints_indicator_map'] = allmaps
        return dic
    def get_plottable_data(self, imgdata):
        ## it is different with base class's method
        # just to be consistent with previous testconvnet.py functions
        # Without transpose and scaling
        ndata = imgdata.shape[-1]
        dimX = imgdata.shape[0]
        res = imgdata.copy() +self.cropped_mean_image.reshape((dimX,1),order='F')
        imgdim = list(self.input_image_dim) + [ndata]
        return res.reshape(imgdim, order='F')
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * 3
        else:
            return iprod(self.depth_joint_indmap.mdim) * self.num_joints
        
class CroppedDHMLPERelSkelJointDataProvider(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        This data provider use relative positions. Each element will be the relative locations w.r.t its parent node
        """
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'

class CroppedDHMLPERelSkelJointPlusDataProvider(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        This data provider use relative positions. Each element will be the relative locations w.r.t its parent node
        This data provider will provide
        [data, joints, joints_indicator_map, joints2d]
        """
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'
    def get_next_batch(self):
        """
        Add 2d joint locations
        """
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        alldata += [np.require(self.data_dic['joints_2d'],dtype=np.single, requirements='C')]
    def get_data_dims(self,idx=0):
        if idx < 3:
            return CroppedDHMLPEJointDataProvider(self,idx)
        else:
            return self.num_joints*2
        
class CroppedDHMLPEPairwiseRelJointDataProvider(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        [imgdata, pairwise_rel_data, indicatormap]
        """
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        # self.feature_name_3d = 'Relative_Y3d_mono_body' <---This is default
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        ## Add pairwise relative location  here
        jdata = np.require(self.data_dic['joints_3d'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')
        alldata += [np.require(dhmlpe_features.calc_pairwise_diff(jdata, 3), dtype=np.single, requirements='C')/self.max_depth]
        ## ADD joint indicator here
        alldata += [np.require(self.data_dic['joints_indicator_map'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * self.num_joints*3
        else:
            return iprod(self.joint_indmap.mdim) * self.num_joints
                
class CroppedDHMLPEJointMaxIndDataProvider(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        dp_params['joint_indmap_type'] = 'maxindmap'
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        
class CroppedDHMLPERelSkelJointMaxIndDataProvider(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        dp_params['joint_indmap_type'] = 'maxindmap'
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'

class CroppedDHMLPERelSkelJointLenDataProvider(CroppedDHMLPERelSkelJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        This data provider will send the length of lim for prediction
        alldata = [imgdata, joint_relskel_3d, indcatormap, lim_length] 
        """
        CroppedDHMLPERelSkelJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)    
    def get_next_batch(self):
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        # Need to add Limb_length_3d
        alldata += [np.require(self.data_dic['Limb_length_3d'], dtype=np.single, requirements='C')/self.max_depth]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        dic = CroppedDHMLPERelSkelJointDataProvider.get_batch(self, batch_num)
        ndata = len(dic['cur_batch_indexes'])
        dic['Limb_length_3d'] = self.batch_meta['Limb_length_3d'][...,dic['cur_batch_indexes']].reshape((-1,ndata),order='F')
        return dic
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * 3
        elif idx == 2:
            return iprod(self.joint_indmap.mdim) * self.num_joints
        else:
            return self.num_joints - 1

class JointDataProvider(CroppedImageDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        This data provider will only provide joint data.
        """
        CroppedImageDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'Relative_Y3d_mono_body'
        self.num_joints = self.batch_meta['num_joints']
        self.max_depth = self.batch_meta['max_depth'] 
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['joints_3d'].shape[-1]
        alldata = [np.require(self.data_dic['joints_3d'].reshape((-1,ndata),order='F')/self.max_depth,dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        """
        batch_num in self.image_range
        """
        dic = dict()
        if self.test and self.shuffle_data == 0:
            # test data doesn't need to circle 
            end_num = min(batch_num + self.batch_size, self.num_image)
            cur_batch_indexes = self.shuffled_image_range[batch_num:end_num]
        else:
            cur_batch_indexes = self.shuffled_image_range[ map(lambda x: x if x < self.num_image else x - self.num_image ,range(batch_num, batch_num + self.batch_size)) ]
        ## record the current batch_indexes
        dic['cur_batch_indexes'] = cur_batch_indexes.copy()
        dic['joints_3d'] = np.concatenate(map(lambda x:self.batch_meta[self.feature_name_3d][...,x].reshape((-1,1),order='F'),dic['cur_batch_indexes']),axis=1)
        return dic
    def get_data_dims(self,idx=0):
        if idx == 0:
            return self.num_joints * 3
        
class RelSkelJointDataProvider(JointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        JointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'Relative_Y3d_mono_body'

class RelSkelJointDoubleDataProvider(RelSkelJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        RelSkelJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.selection_type = 'shuffle'
    def get_next_batch(self):
        epoch, batch_num, alldata = RelSkelJointDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        selected_idx = range(ndata-1,-1,-1)
        if self.selection_type == 'shuffle':
            np.random.shuffle(selected_idx)
        l = []
        for elem in alldata:
            l += [np.require(elem[:,selected_idx].reshape((-1,ndata),order='F'),\
                             dtype=np.single, requirements='C')]
        alldata += l
        return epoch, batch_num, alldata
    def get_data_dims(self,idx=0):
        if idx == 0:
            return RelSkelJointDataProvider.get_data_dims(self,idx)
        else:
            return RelSkelJointDataProvider.get_data_dims(self,idx/2)
# class RelSkelJointDoubleLIndDataProvider(RelSkelJointDoubleDataProvider):
#     def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
#         RelSkelJointDoubleDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
#         self.mpjpe_factor = 300.0 # Need to write to batches.meta in the future
#         self.mpjpe_offset = 0.0
#         self.num_labels = 100
    

class CroppedDHMLPERelSkelPairJt(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        Use RelSkel as targets. And train score function
        [data, joints, joints_indicator_map,
         joints + noise1, score1,
         joints + noise2, score2,
         labels
         ]
        """
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'
        if 'median_sigma' in self.batch_meta:
            self.median_sigma = self.batch_meta['median_sigma']
        else:
            self.median_sigma = 1024
        self.sigma = self.median_sigma / np.sqrt(self.num_joints*3 - 3) / self.max_depth
    @classmethod
    def logistic(cls, x):
        return 1/(1 + np.exp(-x))
    def calc_score(self,x, sigma, num_joints):
        """
        
        """
        s = (sigma**2) * (num_joints - 1) * 3
        score = (1 - self.logistic(np.sum(x**2,0)/s))*2
        return score
    def generate_gauss_noise(self, sigma, ndata):
        """
        No noise on root prediction
        """
        noise = rd.normal(0, sigma, ndata * 3 * self.num_joints).reshape((3*self.num_joints,ndata))
        noise[[0,1,2],:] = 0
        return noise
    def get_next_batch(self):
        """
        generate noise each time
        """
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        ndata = self.data_dic['data'].shape[-1]
        noise1 = self.generate_gauss_noise(self.sigma,ndata)
        score1 = np.require(self.calc_score(noise1, self.sigma, self.num_joints).reshape((1,ndata)), dtype=np.single, requirements='C')
        jt_wn1 = np.require(noise1 + alldata[1], dtype=np.single, requirements='C')
        ##
        noise2 = self.generate_gauss_noise(self.sigma,ndata)
        score2 = np.require(self.calc_score(noise2, self.sigma, self.num_joints).reshape((1,ndata)), dtype=np.single, requirements='C')
        jt_wn2 = np.require(noise2 + alldata[1], dtype=np.single, requirements='C')
        label = (score1 < score2).reshape((1,ndata))
        label = np.require(label, dtype=np.single, requirements='C')
        alldata += [jt_wn1, score1, jt_wn2, score2, label]
        return epoch, batchnum, alldata
    
    def get_data_dims(self,idx=0):
        if idx < 3:
            return CroppedDHMLPEJointDataProvider.get_data_dims(self,idx)
        elif idx == 3:
            return self.num_joints * 3
        elif idx == 5:
            return self.num_joints * 3
        else:
            return 1
    def get_num_classes(self):
        return 2


class CroppedDHMLPERelSkelMixJt(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        Use RelSkel as targets. And train score function
        [data, joints, joints_indicator_map,
         jointsmix, scoremix     <------  curent pose + random noise with prob self.mixprob or
                                          random valid pose with prob 1 - self.mixprob
         ]
        """
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'
        self.mpjpe_offset = 0.0
        self.mpjpe_factor = 300.0 # > 300 will take scores lower than 0.03
        self.sigma = 50.0
        self.mixprob = 0.2
    @classmethod
    def tanh_score(cls, x):
        return 1 - np.tanh(x)
    @classmethod
    def calc_score(cls, z, factor, offset):
        return dutils.calc_tanh_score(z, factor, offset)
    @classmethod
    def calc_mpjpe_from_residual(cls,x, num_joints):
        """
        return 1 x n array
        """
        return dutils.calc_mpjpe_from_residual(x, num_joints)
    @classmethod
    def generate_gauss_noise(cls, num_joints, sigma, ndata):
        """
        No noise on root prediction
        """
        noise = rd.normal(0, sigma, ndata * 3 * num_joints).reshape((3*num_joints,ndata))
        noise[[0,1,2],:] = 0
        return noise
    @classmethod
    def generate_gauss_noise2d(cls, num_joints, sigma, ndata):
        """
        Add noise on each point
        """
        noise = rd.normal(0, sigma, 2 * num_joints*ndata).reshape((2*num_joints,ndata))
        return noise
    @classmethod
    def select_random_pose(cls,pose_array, num_of_sample, index_array):
        """
        randomly select num_of_sample pose with index from index_array and return 
        """
        r = np.minimum(np.floor(np.random.uniform(0,1,num_of_sample) * index_array.size),\
                       index_array.size-1)
        return np.concatenate([pose_array[..., index_array[np.int(k)]].reshape((-1,1), \
                        order='F') for k in r], axis=1)
    def get_next_batch(self):
        """
        generate noise each time
        """
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        ndata = self.data_dic['data'].shape[-1]
        cur_noise = self.generate_gauss_noise(self.num_joints, self.sigma, ndata)/self.max_depth
        jt_withnoise = alldata[1] + cur_noise
        jt = self.select_random_pose(self.batch_meta[self.feature_name_3d], \
                                     ndata, self.shuffled_image_range)/self.max_depth
        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob
        jt[...,jt_mix_ind] = jt_withnoise[...,jt_mix_ind]
        residual = jt - alldata[1]
        mpjpe = self.calc_mpjpe_from_residual(residual, self.num_joints)
        score = self.calc_score(mpjpe, self.mpjpe_factor/self.max_depth, \
                                self.mpjpe_offset/self.max_depth).reshape((1,ndata))
        alldata += [np.require(jt, dtype=np.single, requirements='C'),\
                    np.require(score,dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    
    def get_data_dims(self,idx=0):
        if idx < 3:
            return CroppedDHMLPEJointDataProvider.get_data_dims(self,idx)
        elif idx in  [3]:
            return self.num_joints * 3
        else:
            return 1
    def get_num_classes(self):
        return 2
class CroppedDHMLPERelSkelMixJtPlus(CroppedDHMLPERelSkelMixJt):
    """
    This dataprovider will add 2d candidates
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelMixJt.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_2d_factor = 25.0
        self.mpjpe_2d_offset = 0.0
        self.sigma_2d = 7
        self.mixprob2d = 0.2
    def get_next_batch(self):
        epoch, batchnum, alldata = CroppedDHMLPERelSkelMixJt.get_next_batch(self)
        ndata = self.data_dic['data'].shape[-1]
        ## generate 2d "candidate pose"
        cur_noise_2d = self.generate_gauss_noise2d(self.num_joints, self.sigma_2d, ndata)
        jt2d_withnoise = self.data_dic['joints_2d'] + cur_noise_2d
        offset_2d = np.tile(np.concatenate([np.asarray(self.cur_offset_c).reshape((1,-1)), np.asarray(self.cur_offset_r).reshape((1,-1))],axis=0), [self.num_joints,1])
        jt2d = self.select_random_pose(self.batch_meta['Y2d_bnd_body'], \
                                     ndata, self.shuffled_image_range)
        jt2d = jt2d - offset_2d
        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob2d
        jt2d[...,jt_mix_ind] = jt2d_withnoise[...,jt_mix_ind]
        residual = jt2d - self.data_dic['joints_2d']
        mpjpe2d = self.calc_mpjpe_from_residual(residual, self.num_joints)
        score2d = self.calc_score(mpjpe2d, self.mpjpe_2d_factor, \
                                  self.mpjpe_2d_offset).reshape((1,ndata))
        # for t in [0.1,0.2,0.3,0.5,0.7,0.9]:
        #     print '%.6f < %.6f' % (np.sum(score2d < t)*100.0/score2d.size, t)
        # print '============='
        #normalize 2d data into [0,1]
        jt2d = np.require(jt2d,dtype=np.single) / np.tile(np.asarray([self.input_image_dim[1], \
                                            self.input_image_dim[0]]).reshape((2,1)),
                                            [self.num_joints, 1])
        alldata += [np.require(jt2d, dtype=np.single, requirements='C'),\
                    np.require(score2d, dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx < 5:
            return CroppedDHMLPERelSkelMixJt.get_data_dims(self,idx)
        elif idx == 5:
            return self.num_joints * 2
        else:
            return 1
class CroppedDHMLPERelSkelMixJtLabel(CroppedDHMLPERelSkelMixJt):
    """
    This DP add labels for scoremix
    The default is 100 classes
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelMixJt.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.num_labels = 100.0
    def get_next_batch(self):
        """
        
        """
        epoch, batchnum, alldata = CroppedDHMLPERelSkelMixJt.get_next_batch(self)
        label = np.minimum(np.floor(alldata[-1] * self.num_labels),self.num_labels-1)
        alldata += [np.require(label.reshape(1,-1),dtype=np.single,requirements='C')]
        return epoch,batchnum, alldata
    def get_num_classes(self):
        return self.num_labels

class CroppedDHMLPERelSkelMixJtMultiLabel(CroppedDHMLPERelSkelMixJt):
    """
    This DP add labels for scoremix
    This label is the a vector of indicators
    The default is 100 classes
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelMixJt.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.num_labels = np.int(100)
    def get_next_batch(self):
        """
        
        """
        epoch, batchnum, alldata = CroppedDHMLPERelSkelMixJt.get_next_batch(self)
        label = np.minimum(np.floor(alldata[-1] * self.num_labels),self.num_labels-1)
        template = np.eye(self.num_labels)
        ndata = label.size
        ind = template[...,np.require(label.flatten(),dtype=np.int)].reshape((self.num_labels, ndata),order='F')
        alldata += [np.require(ind,dtype=np.single,requirements='C')]
        return epoch,batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx < 5:
            return CroppedDHMLPERelSkelMixJt.get_data_dims(self,idx)
        else:
            return self.num_labels
    def get_num_classes(self):
        return self.num_labels


class CroppedDHMLPERelSkelMixJtLabel20(CroppedDHMLPERelSkelMixJtLabel):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelMixJtLabel.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.num_labels = 20
class CroppedDHMLPERelSkelMixJtLabel10(CroppedDHMLPERelSkelMixJtLabel):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelMixJtLabel.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.num_labels = 10

        
class CroppedDHMLPERelSkelJointDoubleLIndDataProvider(CroppedDHMLPEJointDoubleDataProvider):
    """
    This data provider will provide
    [data, joints, joints_indicator_map,
    _data, _joints, _joints_indicator_map,
    scores, labels,ind]
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPEJointDoubleDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_factor = 300.0 # Need to write to batches.meta in the future
        self.mpjpe_offset = 0.0
        self.num_labels = 100
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'
    @classmethod
    def calc_score(cls, z, factor, offset):
        return dutils.calc_tanh_score(z,factor, offset)
    def get_next_batch(self):
        epoch, batch_num, alldata = CroppedDHMLPEJointDoubleDataProvider.get_next_batch(self)
        residuals = alldata[1] - alldata[4]
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, self.num_joints)
        ndata = alldata[0].shape[-1]
        scores = self.calc_score(mpjpe, self.mpjpe_factor/self.max_depth, \
                                 self.mpjpe_offset/self.max_depth).reshape((1,ndata))
        labels = np.minimum(np.floor(scores * self.num_labels),self.num_labels-1)
        # for t in np.linspace(0,1,7):
        #     print 'There %.6f%% < %.6f' % (np.sum(scores < t)*100.0/scores.size, t)
        template = np.eye(np.int(self.num_labels))
        ind = template[...,np.require(labels.flatten(),dtype=np.int)].reshape((np.int(self.num_labels), ndata),order='F')
        alldata += [np.require(scores.reshape((1,ndata)),dtype=np.single, requirements='C'), \
                    np.require(labels.reshape((1,ndata)),dtype=np.single, requirements='C'),\
                    np.require(ind,dtype=np.single, requirements='C')]
        return epoch, batch_num, alldata
    def get_data_dims(self,idx=0):
        if idx < 6:
            return CroppedDHMLPEJointDoubleDataProvider.get_data_dims(self,idx)
        elif idx == 8:
            return np.int(self.num_labels)
        else:
            return 1
    def get_num_classes(self,idx=0):
        return np.int(self.num_labels)


class MemoryFeatureRandDataProvider(MemoryFeatureDataProvider):
    """
    This data provider take all the data field from the memory data provider
    and it will add random data in the last field
    The only requirements is the element in random_feature_list will have the same
    last dimension (number of data).
    """
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        MemoryFeatureDataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.random_feature_dim = self.batch_meta['random_feature_dim']
        if len(self.random_feature_dim) != len(self.batch_meta['random_feature_list']):
            raise DHMLPEDataProviderError('Inconsistent data dimension %d (dim_list) vs %d (random feature_list)' % (len(self.random_feature_list), len(self.batch_meta['random_feature_list']))) 
    def get_next_batch(self):
        epoch, batchnum, alldata = MemoryFeatureDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        for i, elem in enumerate(self.batch_meta['random_feature_list']):
            selection_idx_t = np.random.choice(self.num_feature, ndata)
            selection_idx = self.feature_range[selection_idx_t]
            cur_feature = np.require(elem[...,selection_idx].reshape((-1,ndata),order='F'), \
                                     dtype=np.single, requirements='C')
            alldata += [cur_feature]
        return epoch, batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx < len(self.feature_dim):
            return self.feature_dim[idx]
        else:
            return self.random_feature_dim[idx - len(self.feature_dim)]
            
class MemoryJointPredictionMixDataProvider(MemoryFeatureRandDataProvider):
    """
    Use this with care.
    I assume the following data structure,
    feature_list
        joint_prediction
    random_list
        joint_ground_truth (after dividing self.max_depth)

    The output of this data provider is
        joint_prediction, mix_joint, score
    Please note the the score /mpjpe is based on relskel
    """
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        MemoryFeatureRandDataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_offset = 0.0
        self.mpjpe_factor = 300.0 # > 300 will take scores lower than 0.03
        self.sigma = 50.0
        self.mixprob = 0.2
        self.max_depth = self.batch_meta['info']['max_depth']
        self.num_joints = self.batch_meta['info']['num_joints']
        if len(self.batch_meta['feature_list'])!=1 or \
          len(self.batch_meta['random_feature_list'])!=1:
            raise DHMLPEDataProviderError('the dimension of (random) feature_list) should be 1')
    @classmethod
    def calc_score(cls,z, factor, offset):
        return dutils.calc_tanh_score(z, factor, offset)
    def get_next_batch(self):
        epoch, batchnum, alldata = MemoryFeatureRandDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        cur_noise = dutils.generate_gauss_noise(self.num_joints, self.sigma, ndata, dim=3, ignore_root=True)/self.max_depth
        gt_pose = self.batch_meta['random_feature_list'][0][...,self.data_dic['cur_batch_indexes']].reshape((-1,ndata),order='F')
        jt_withnoise = gt_pose + cur_noise
        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob
        mix = alldata[1]
        mix[...,jt_mix_ind] = jt_withnoise[...,jt_mix_ind]
        residuals = mix - gt_pose
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, self.num_joints)
        scores = self.calc_score(mpjpe, self.mpjpe_factor/self.max_depth, \
                                 self.mpjpe_offset/self.max_depth).reshape((1,ndata))
        # for t in np.linspace(0,1, 10):
        #     print '%.6f%% < %.6f' % (np.sum(scores<t)*100.0/scores.size, t)
        alldata += [np.require(scores, dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if (idx < 2):
            return MemoryFeatureRandDataProvider.get_data_dims(self,idx)
        else:
            return 1
        
class MemoryJointPredictionAngleMixDataProvider(MemoryFeatureRandDataProvider):
    """
    This data provider will add noise on angle instead of feature
        I assume the following data structure,
    feature_list
        joint_prediction
    random_list
        joint_ground_truth (after dividing self.max_depth)

    The output of this data provider is
    joint_prediction, mix_angle_joint, score

    The score is still defined in terms of relskel
    """
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        import iread.h36m_hmlpe as h36m
        MemoryFeatureRandDataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_offset = 0.0
        self.mpjpe_factor = 300.0 # > 300 will take scores lower than 0.03
        self.mixprob = 0.75
        self.gmm_prob = [0.2] * 5
        self.sigma = [20, 10, 5, 2.5, 1.25] # This is for the noise of angle data
        self.max_depth = self.batch_meta['info']['max_depth']
        self.num_joints = self.batch_meta['info']['num_joints']
        if len(self.batch_meta['feature_list'])!=1 or \
          len(self.batch_meta['random_feature_list'])!=1:
            raise DHMLPEDataProviderError('the dimension of (random) feature_list) should be 1')
        self.angles = self.batch_meta['info']['mono_angles']
        self.skel = self.batch_meta['info']['skel']
        self.cutils = dutils.Cutils()

        self.subject_ids = self.batch_meta['info']['h36m_id'][0,:].flatten(order='F')
        self.angle_dim = self.angles.shape[0]
        self.angle_useful_idx = self.batch_meta['info']['angle_useful_idx']
        self.part_idx = h36m.part_idx
        self.body_idx = h36m.body_idx
    def angle2pose(self, angle, subject_ids):
        """
        Just ensure they are the feature of the data
        """
        # import cutils
        # ndata = angle.shape[-1]
        # skel = np.require(self.skel.flatten(order='F'),dtype=np.float64)
        # angle = np.require(angle.flatten(order='F'),dtype=np.float64)
        # subject_ids = np.require(subject_ids.flatten(order='F'),dtype=np.int)
        # return cutils.convert_angle2pose([[skel.tolist(), angle.tolist(), subject_ids.tolist()],[np.int(32),np.int(8),ndata, np.int(78)]])
        return self.cutils.convert_angle2pose(self.skel.copy(), angle, subject_ids)
    @classmethod
    def selective_add(cls, X, Y, dims):
        res = X.copy()
        res[dims,:] = X[dims,:] + Y[dims,:]
        return res
    def calc_relskel(self, x):
        """
        x is dim_coor x num_joints x ndata
        """
        r = x.copy()
        ndata = x.shape[-1]
        r[:,[t[1] for t in self.part_idx],:] -= x[:,[t[0] for t in self.part_idx],:]
        r[:,0,:] = 0
        return r.reshape((-1, self.num_joints,ndata),order='F')
    @classmethod
    def calc_score(cls,z, factor, offset):
        return dutils.calc_tanh_score(z, factor, offset)

    def get_next_batch(self):
        epoch, batchnum, alldata = MemoryFeatureRandDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        
        cur_noise = dutils.generate_GMM_noise(self.angle_dim, self.sigma, self.gmm_prob, ndata, dim=1, ignore_root=False).reshape((-1,ndata),order='F')        
        gt_pose = self.batch_meta['random_feature_list'][0][...,self.data_dic['cur_batch_indexes']].reshape((-1,ndata),order='F')
        gt_angle = self.angles[...,self.data_dic['cur_batch_indexes']].reshape((-1,ndata),order='F')
        cur_subject_ids = self.subject_ids[self.data_dic['cur_batch_indexes']]
        angle_with_noise = self.selective_add(gt_angle, cur_noise, self.angle_useful_idx)
        poses = self.angle2pose(angle_with_noise, cur_subject_ids)

        poses = poses.reshape((3,-1,ndata),order='F') # This is determined
        poses = poses[:,self.body_idx,:]
        rel_poses = self.calc_relskel(poses).reshape((-1,ndata),order='F')/self.max_depth

        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob
        mix = alldata[1]

        mix[...,jt_mix_ind] = rel_poses[...,jt_mix_ind]
        
        residuals = mix - gt_pose
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, self.num_joints)
        scores = self.calc_score(mpjpe, self.mpjpe_factor/self.max_depth, \
                                 self.mpjpe_offset/self.max_depth).reshape((1,ndata))
        # for t in np.linspace(0,1, 10):
        #     print '%.6f%% < %.6f' % (np.sum(scores<t)*100.0/scores.size, t)
        alldata[1] = np.require(mix, dtype=np.single, requirements='C')
        alldata += [np.require(scores, dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if (idx < 2):
            return MemoryFeatureRandDataProvider.get_data_dims(self,idx)
        else:
            return 1
 
class MemoryJointPredictionKNNMixDataProvider(MemoryFeatureDataProvider):
    """
    Use this with care.
    I assume the following data structure,
    feature_list
        joint_prediction
    random_list
        joint_ground_truth (after dividing self.max_depth)

    The output of this data provider is
        joint_prediction, mix_joint, score
    """
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        MemoryFeatureDataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_offset = 0.0
        self.mpjpe_factor = 150.0 # > 300 will take scores lower than 0.03
        self.sigma = 5.0
        self.mixprob = 0.9
        self.max_depth = self.batch_meta['info']['max_depth']
        self.num_joints = self.batch_meta['info']['num_joints']
        if len(self.batch_meta['feature_list'])!=1 or \
          len(self.batch_meta['random_feature_list'])!=1:
            raise DHMLPEDataProviderError('the dimension of (random) feature_list) should be 1')
        mpjpe_train_all = self.batch_meta['info']['mpjpe'][..., self.batch_meta['info']['train_range']]
        score_train = self.calc_score(mpjpe_train_all, self.mpjpe_factor, self.mpjpe_offset) 
        avg_score = np.mean(score_train, axis=1)
        m_score = np.min(avg_score.flatten())
        self.score_offset = max(0, m_score - 2e-2)
        sample_prob = dutils.calc_sample_prob(avg_score, self.score_offset, reverse=True)
        self.sample_prob = dutils.smooth_prob(sample_prob, 0.02)
        self.knn_index = self.batch_meta['info']['knn_index']
        self.knn_K = self.knn_index.shape[0]
    @classmethod
    def calc_score(cls,z, factor, offset):
        return dutils.calc_tanh_score(z, factor, offset)
    def get_next_batch(self):
        epoch, batchnum, alldata = MemoryFeatureDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        gt_rel_pose = self.batch_meta['info']['gt_relative_mono'][...,self.data_dic['cur_batch_indexes']].reshape((-1,ndata),order='F')

        KNN_sample_index = np.random.choice(self.knn_K, ndata, replace=True, \
                                            p=self.sample_prob) 
        index_in_train = [self.knn_index[KNN_sample_index[k],\
                                         self.data_dic['cur_batch_indexes'][k]] \
                          for k in range(ndata)]
        knn_pose = self.batch_meta['info']['gt_relative_mono'][..., index_in_train].reshape((-1,ndata),order='F')
        
        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob

        #generate random poses
        selection_idx_t = np.random.choice(self.num_feature, ndata)
        selection_idx = self.feature_range[selection_idx_t]
        mix = self.batch_meta['info']['gt_relative_mono'][..., selection_idx].reshape((-1,ndata),order='F')
            
        mix[...,jt_mix_ind] = knn_pose[...,jt_mix_ind]
        mix_noise = mix + dutils.generate_gauss_noise(self.num_joints, self.sigma, ndata, 3)/self.max_depth
        mix_noise_relskel = dutils.calc_relskel(mix_noise, self.num_joints).reshape((-1,ndata),order='F')
        residuals = mix_noise - gt_rel_pose
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, self.num_joints)
        scores = self.calc_score(mpjpe, self.mpjpe_factor/self.max_depth, \
                                 self.mpjpe_offset/self.max_depth).reshape((1,ndata))
        # for t in np.linspace(0,1, 10):
        #     print '%.6f%% < %.6f' % (np.sum(scores<t)*100.0/scores.size, t)
        alldata += [np.require(mix_noise_relskel, dtype=np.single, requirements='C')]
        alldata += [np.require(scores, dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if (idx < 1):
            return MemoryFeatureDataProvider.get_data_dims(self,idx)
        elif idx == 1:
            return self.num_joints * 3
        else:
            return 1
 
class MemoryJointPredictionKNNMixLIndDataProvider(MemoryJointPredictionKNNMixDataProvider):
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        return labels, ind
        """
        MemoryJointPredictionKNNMixDataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.num_labels = np.int(10)
        self.template = np.eye(self.num_labels)
    def get_next_batch(self):
        epoch, batchnum, alldata = MemoryJointPredictionKNNMixDataProvider.get_next_batch(self)
        scores = alldata[-1].flatten()
        ndata = alldata[0].shape[-1]
        labels = np.require(np.minimum(np.floor(scores * self.num_labels), \
                                       self.num_labels -1).reshape((1,ndata),order='F'),\
                                        dtype=np.single, requirements='C')
        ind = np.require( self.template[..., np.require(labels.flatten(),dtype=np.int)], \
                          dtype=np.single, requirements='C')
        alldata += [labels, ind]
        return epoch, batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx < 3:
            return MemoryJointPredictionKNNMixDataProvider.get_data_dims(self,idx)
        elif idx == 4:
            return self.num_labels
        
class MemoryJointPredictionKNNMix_T_DataProvider(MemoryJointPredictionKNNMixDataProvider):
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        return labels, ind
        """
        MemoryJointPredictionKNNMixDataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_factor = 300.0
        self.mixprob = 0.25
class CroppedDHMLPERelSkelJointRandJtDataProvider(CroppedDHMLPERelSkelJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        """
        This data provider use relative positions. Each element will be the relative locations w.r.t its parent node
        [
        image_data, joints, indicator map
         random joints, scores
        ]
        Please note that the score calculation here is based on relskel.
        Just to follow MemoryJointPredictionMixDataProvider
        """
        CroppedDHMLPERelSkelJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.mpjpe_factor = 300.0
        self.mpjpe_offset = 0.0
        self.sigma = 10.0
        self.mixprob = 0.2
        self.gt_pose = self.batch_meta['Relative_Y3d_mono_body']
        # print 'feature_name_3d is' , self.feature_name_3d
    def get_next_batch(self):
        # import iutils as iu
        # import iread.myio as mio
        # cache_name = '/opt/visal/tmp/for_sijin/tmp/cahches/buffer_nov_10'
        # if iu.exists(cache_name, 'file'):
        #     d = mio.unpickle(cache_name)
        #     epoch,batchnum, alldata = d['epoch'], d['batchnum'], d['alldata']
        #     self.data_dic = d['data_dic']
        # else:
        #     epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        #     d = dict()
        #     d['epoch'], d['batchnum'], d['alldata'] = epoch,batchnum, alldata
        #     d['data_dic'] = self.data_dic
        #     mio.pickle(cache_name, d)
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        selected_idx = np.random.choice(self.shuffled_image_range, ndata)
        mix = self.gt_pose[..., selected_idx].reshape((-1,ndata),order='F')
        gt_pose = self.gt_pose[..., self.data_dic['cur_batch_indexes']].reshape((-1,ndata),order='F')
        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob
        mix[..., jt_mix_ind] = gt_pose[..., jt_mix_ind]
        cur_noise = dutils.generate_gauss_noise(self.num_joints, self.sigma, ndata, dim=3, ignore_root=True)
        mix = (mix + cur_noise)
        rel_mix = dutils.calc_relskel(mix, self.num_joints).reshape((-1,ndata),order='F')
        residuals = dutils.calc_relskel(gt_pose, self.num_joints).reshape((-1,ndata),order='F') - rel_mix
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, self.num_joints)
        scores = dutils.calc_tanh_score(mpjpe, self.mpjpe_factor, \
                                 self.mpjpe_offset).reshape((1,ndata), order='F')
        alldata+= [ np.require(rel_mix/self.max_depth, dtype=np.single, requirements='C'), \
                    np.require(scores, dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
                    
    def get_data_dims(self,idx=0):
        if idx < 3:
            return CroppedDHMLPERelSkelJointDataProvider.get_data_dims(self, idx)
        elif idx == 3:
            return self.num_joints * 3
        else:
            return 1

class CroppedDHMLPERelSkelJointRandPairJtDataProvider(CroppedDHMLPERelSkelJointDataProvider):
    """
    This data provider use relative skel as feature.
    [
    image_data, joints, indicator map
    random_joint_0, random_joint_1
    label                         <---- 0 if 0 is closer to gt, 1 otherwise
    ]
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.mixprob = 0.2
        self.gt_pose = self.batch_meta['Relative_Y3d_mono_body']
        self.sigma = 5
    def get_random_poses(self, ndata, cur_batch_indexes, cur_gt_pose):
        selected_idx = np.random.choice(self.shuffled_image_range, ndata)
        mix = self.gt_pose[..., selected_idx].reshape((-1,ndata),order='F')
        jt_mix_ind = np.random.uniform(0,1,ndata) < self.mixprob
        mix[..., jt_mix_ind] = cur_gt_pose[..., jt_mix_ind]
        cur_noise = dutils.generate_gauss_noise(self.num_joints, self.sigma, ndata, dim=3, ignore_root=True)
        mix = (mix + cur_noise)
        return mix
    def get_next_batch(self):
        # import iutils as iu
        # import iread.myio as mio
        # cache_name = '/opt/visal/tmp/for_sijin/tmp/cahches/buffer_nov_10'
        # if iu.exists(cache_name, 'file'):
        #     d = mio.unpickle(cache_name)
        #     epoch,batchnum, alldata = d['epoch'], d['batchnum'], d['alldata']
        #     self.data_dic = d['data_dic']
        # else:
        #     epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        #     d = dict()
        #     d['epoch'], d['batchnum'], d['alldata'] = epoch,batchnum, alldata
        #     d['data_dic'] = self.data_dic
        #     mio.pickle(cache_name, d)
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        cur_batch_indexes = self.data_dic['cur_batch_indexes']
        gt_pose = self.gt_pose[..., cur_batch_indexes].reshape((-1,ndata),order='F')
        
        mix_0 = self.get_random_poses(ndata, cur_batch_indexes, gt_pose)
        mix_1 = self.get_random_poses(ndata, cur_batch_indexes, gt_pose)
        residuals_0 = gt_pose - mix_0
        residuals_1 = gt_pose - mix_1
        mpjpe_0 = dutils.calc_mpjpe_from_residual(residuals_0, self.num_joints)
        mpjpe_1 = dutils.calc_mpjpe_from_residual(residuals_1, self.num_joints)
        label = mpjpe_0 < mpjpe_1
        template = np.eye(2)
        ind = template[:,np.asarray(label.flatten(),dtype=np.int)]

        rel_mix_0 = dutils.calc_relskel(mix_0, self.num_joints).reshape((-1,ndata),order='F')
        rel_mix_1 = dutils.calc_relskel(mix_1, self.num_joints).reshape((-1,ndata),order='F')
        print '%.6f %% of the cases prefer 0' % (np.sum(label ==0)*100.0/label.size)
        alldata+= [ np.require(rel_mix_0/self.max_depth, dtype=np.single, requirements='C'), \
                    np.require(rel_mix_1/self.max_depth, dtype=np.single, requirements='C'), \
                    np.require(label.reshape((1,ndata)), dtype=np.single, requirements='C'), \
                    np.require(ind, dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx < 3:
            return CroppedDHMLPERelSkelJointDataProvider.get_data_dims(self, idx)
        elif idx in [3,4]:
            return self.num_joints * 3
        elif idx == 6:
            return 2
        else:
            return 1
