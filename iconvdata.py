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

from data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
#CONV_IMG_SIZE = 128 # 128 for the whole image. Attention
#CONV_IMG_SIZE=56
CONV_IMG_SIZE=112

class POSEDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((-1, d['data'].shape[1]), order='F'), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        alldata = [ datadic['data']] + [ n.require(l.reshape((1,datadic['data'].shape[1]), order='F'), dtype=n.single, requirements='C') for l in datadic['labels']]
        return epoch, batchnum, alldata

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3) / 255.0, dtype=n.single)


class LargeMultiPOSEDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE

    def get_next_batch(self):
        self.data_dic = self.get_batch(self.curr_batchnum)
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['labels'] = n.c_[n.require(self.data_dic['labels'], dtype=n.single)].reshape((-1, self.data_dic['data'].shape[1]), order='F')
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        alldata = [ self.data_dic['data'] ] + [n.require(l.reshape((1,self.data_dic['data'].shape[1]), order='F'), dtype=n.single, requirements='C') for l in self.data_dic['labels']]    
        return epoch, batchnum, alldata

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
       return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3) / 255.0, dtype=n.single)



class MultiPOSEDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((-1, d['data'].shape[1]), order='F'), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        alldata = [ datadic['data'] ] + [n.require(l.reshape(1,datadic['data'].shape[1]), requirements='C') for l in datadic['labels']]        
        return epoch, batchnum, alldata

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
       return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3) / 255.0, dtype=n.single)


class LargeJoints8DataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE
    def get_num_parts(self):
        return 8
    """
    In this version, 
    all nan value are removed 
    so that all the data point is valid
    """
    def get_next_batch(self):
        self.data_dic = self.get_batch(self.curr_batchnum)
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['joints8'] = n.require(self.data_dic['joints8'].reshape((16, self.data_dic['data'].shape[1]), order='C'), dtype=n.single, requirements='C')
        valid_idx = n.require(1 - n.max(n.isnan(self.data_dic['joints8']), axis=0), dtype=n.bool)
        self.data_dic['data'] = n.require(self.data_dic['data'][...,valid_idx], requirements='C')        
        self.data_dic['joints8'] = n.require(self.data_dic['joints8'][...,valid_idx],dtype=n.single, requirements='C')
        self.data_dic['joints8'] = self.data_dic['joints8']/self.img_size;
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        alldata = [self.data_dic['data'], self.data_dic['joints8']]  
        
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
       return self.img_size**2 * self.num_colors if idx == 0 else 16
    def get_plottable_data(self, data):
        return n.require(data + self.data_mean).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
    def get_joints(self):
        return self.data_dic['joints8'].reshape((8,2,-1), order='C')
    def get_num_classes(self):
        return 10


class LargeJoints8AndLabelDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE
    def get_num_parts(self):
        return 8
    """
    In this version, 
    all nan value are removed 
    so that all the data point is valid
    """
    def get_next_batch(self):
        self.data_dic = self.get_batch(self.curr_batchnum)
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['joints8'] = n.require(self.data_dic['joints8'].reshape((16, self.data_dic['data'].shape[1]), order='C'), dtype=n.single, requirements='C')
        self.data_dic['labels'] = n.require(self.data_dic['labels'].reshape((-1, self.data_dic['data'].shape[1]), order='F'), dtype=n.single, requirements='C') 
        #change_time = time()
        valid_idx = n.require(1 - n.max(n.isnan(self.data_dic['joints8']), axis=0), dtype=n.bool)
        self.data_dic['data'] = n.require(self.data_dic['data'][...,valid_idx], requirements='C')        
        self.data_dic['joints8'] = n.require(self.data_dic['joints8'][...,valid_idx],dtype=n.single, requirements='C')
        self.data_dic['labels'] = n.require(self.data_dic['labels'][...,valid_idx], dtype=n.single, requirements='C')
        #print 'change time to %.3f' % (time() - change_time)
        # normalize joints to [0,1]
        self.data_dic['joints8'] = self.data_dic['joints8']/self.img_size;
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        alldata = [self.data_dic['data'], self.data_dic['joints8']]
        # note that all label data should have offset 2
        alldata = alldata + [n.require(l.reshape((1,self.data_dic['data'].shape[1]), order='F'), dtype=n.single, requirements='C') for l in self.data_dic['labels']]
        
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_size**2 * self.num_colors
        elif idx == 1:
            return 16
        else:
            return 1
    def get_plottable_data(self, data):
        return n.require(data + self.data_mean).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
    def get_joints(self):
        return self.data_dic['joints8'].reshape((8,2,-1), order='C')
    def get_num_classes(self):
        return 10

class LargeJoints8AndLabelAllDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE
    def get_num_parts(self):
        return 8
    """
    In this version, 
    all nan value are removed 
    so that all the data point is valid
    """
    def get_next_batch(self):
        load_time = time()
        self.data_dic = self.get_batch(self.curr_batchnum)
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['joints8'] = n.require(self.data_dic['joints8'].reshape((16, self.data_dic['data'].shape[1]), order='C'), dtype=n.single, requirements='C')
        self.data_dic['labels'] = n.require(self.data_dic['labels'].reshape((-1, self.data_dic['data'].shape[1]), order='F'), dtype=n.single, requirements='C') 
        self.data_dic['joints8'] = self.data_dic['joints8']/self.img_size;
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        alldata = [self.data_dic['data'], self.data_dic['joints8']]
        # note that all label data should have offset 2
        alldata = alldata + [n.require(l.reshape((1,self.data_dic['data'].shape[1]), order='F'), dtype=n.single, requirements='C') for l in self.data_dic['labels']]
        #print 'Loading data takes %.3f\n' % ( time() - load_time)
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_size**2 * self.num_colors
        elif idx == 1:
            return 16
        else:
            return 1
    def get_plottable_data(self, data):
        return n.require(data + self.data_mean).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
    def get_joints(self):
        return self.data_dic['joints8'].reshape((8,2,-1), order='C')
    def get_num_classes(self):
        return 10

class LargeJoints8AndIndicatorAllDataProvider(DataProvider):
        def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
                DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
                self.data_mean = self.batch_meta['data_mean']
                self.num_colors = 3
                self.img_size = CONV_IMG_SIZE
                if 'indmap_para' not in self.batch_meta:
                    self.ind_dim = 7 * 8 * 8
                else:
                    self.ind_dim = self.batch_meta['indmap_para']['dim']
                # if it is set to be True, get_next_batch
                # get_next_batch will not load new dic  
                self.keep_dic = False
        def set_data_dic(self, dic):
                self.data_dic = dic
        def get_next_batch(self):
                if not self.keep_dic:
                    self.data_dic = self.get_batch(self.curr_batchnum)
                imgdata = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
                self.data_dic['joints8'] = n.require(self.data_dic['joints8'].reshape((16, self.data_dic['data'].shape[1]), order='C'), dtype=n.single, requirements='C')
                self.data_dic['joints8'] = self.data_dic['joints8']/self.img_size;
                epoch, batchnum = self.curr_epoch, self.curr_batchnum
                self.advance_batch()
                alldata = [imgdata, self.data_dic['joints8'].copy()]
                ind = self.data_dic['indmap']
                # Note that all image like data will use 'F' order
                # although it will require 'C' continuous data
                # 0th body parts will start as idx 2
                ndata = ind.shape[-1]
                self.data_dic['indmap'] = n.require(ind.reshape((-1, ndata), order='F'), dtype=n.single, requirements='C')
                alldata += [self.data_dic['indmap']  ]
                self.ind_dim = ind.shape[0] * ind.shape[1] * ind.shape[2]
                return epoch, batchnum, alldata
        def get_data_dims(self, idx=0):
                if idx == 0:
                    return self.img_size**2 * self.num_colors
                elif idx == 1:
                    return 16
                else:# should be called after get_next_batch
                    return self.ind_dim
        def get_plottable_data(self, data):
                return n.require(data + self.data_mean).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
        def get_joints(self):
                return self.data_dic['joints8'].reshape((8,2,-1), order='C')
        def get_num_parts(self):
                return 7
class LargeJtInd2_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    """
    This DataProvider will add joint indicator at the last position in alldata
    In this version, two indicator map(part,joint) are required to be the same size
    """
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LargeJoints8AndIndicatorAllDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        if 'savedata_info' in self.batch_meta and 'jt_inddim' in self.batch_meta['savedata_info']['indmap_para']:
            self.jt_inddim = self.batch_meta['savedata_info']['indmap_para']['jt_inddim']
        else:
            self.jt_inddim = 8 * 8 * 8
        
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        ind = self.data_dic['joint_indmap']
        self.data_dic['joint_indmap'] = n.require(ind.reshape((-1,ndata),order='F'), dtype=n.single, requirements='C')
        alldata += [self.data_dic['joint_indmap']]
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if idx <3:
            return LargeJoints8AndIndicatorAllDataProvider.get_data_dims(self, idx)
        else:
            return self.jt_inddim

class LargeJtInd2Mask_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    """
    The content in alldata will be
    0 : imgdata,   1:  joints8,   2:  part_indicator_map,
    3 : joint_indicator_map,      4: joint_mask  5:is_positive 
    """
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LargeJoints8AndIndicatorAllDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        prod = lambda x: x[0] * x[1]
        #nparts = self.batch_meta['nparts']
        njoints = self.batch_meta['njoints']
        #self.pt_inddim = prod(self.batch_meta['ind_dim']['part_indmap'])*nparts   
        self.jt_inddim = prod(self.batch_meta['ind_dim']['joint_indmap'])*njoints
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        ind = self.data_dic['joint_indmap']
        self.data_dic['joint_indmap'] = n.require(ind.reshape((-1,ndata),order='F'), dtype=n.single, requirements='C')
        alldata += [self.data_dic['joint_indmap']]
        mask = self.data_dic['jointmasks'].reshape((-1,ndata),order='C') 
        self.data_dic['jointmasks'] = n.require(mask,dtype=n.single,requirements='C')
        # change nan to zero in joints8
        alldata[1][~n.require(mask,dtype=n.bool)] = 0
        alldata += [self.data_dic['jointmasks']]
        is_positive = n.require(self.data_dic['is_positive'].reshape((-1,ndata),order='F'),dtype=n.single, requirements='C')
        self.data_dic['is_positive'] = is_positive
        alldata += [is_positive]
        return epoch, batchnum, alldata
    def get_data_dims(self,idx=0):
        if idx < 3:
            return LargeJoints8AndIndicatorAllDataProvider.get_data_dims(self,idx)
        elif idx == 3:
            return self.jt_inddim
        elif idx == 4:
            return LargeJoints8AndIndicatorAllDataProvider.get_data_dims(self,1)
        else:
            return 1                
               
class LargeJtIndLack_LUA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #LUA is 5-th part
        mask[5,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata
class LargeJtIndLack_RUA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #RUA is 5-th part
        mask[5,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata

class LargeJtIndLack_RLA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #RLA is 6-th part
        mask[6,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata    
class LargeJtIndLack_LUA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #LUA is 5-th part
        mask[3,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata

class LargeJtIndLack_LLA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #RLA is 6-th part
        mask[4,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata    
class LargeJtIndLack_UA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #RUA is 5-th part, LUA is 3-th part
        mask[5,:,:] = 0
        mask[3,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata
class LargeJtIndLack_LA_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #RLA is 6-th part, LLA is 4-th part
        mask[6,:,:] = 0
        mask[4,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata
class LargeJtIndLack_HEAD_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #Head is 0-th part
        mask[0,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata
class LargeJtIndLack_SHOULDER_DataProvider(LargeJoints8AndIndicatorAllDataProvider):
    def get_next_batch(self):
        epoch, batchnum, alldata = LargeJoints8AndIndicatorAllDataProvider.get_next_batch(self)
        ndata = alldata[0].shape[-1]
        num_parts = self.get_num_parts()
        ind_dim = self.get_data_dims(2)
        mask = n.ones((num_parts, ind_dim/num_parts, ndata),dtype=n.single, order='F')
        #Head is 0-th part
        mask[1,:,:] = 0
        mask[2,:,:] = 0
        alldata += [n.require(mask.reshape((-1,ndata), order='F'), requirements='C')]
        return epoch, batchnum, alldata

class LargeJoints8AndIndicatorMaskAllDataProvider(DataProvider):
    """
    This data provider is used for providing various masks for network
    dataidx = 0: imgdata
              1: joints8
              2: indmap
              3: mask0
              ... 
    """
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = CONV_IMG_SIZE
        if 'indmap_para' not in self.batch_meta:
            self.ind_dim = 7 * 8 * 8
        else:
            self.ind_dim = self.batch_meta['indmap_para']['dim']
    def get_next_batch(self):
        load_time = time()
        self.data_dic = self.get_batch(self.curr_batchnum)
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['joints8'] = n.require(self.data_dic['joints8'].reshape((16, self.data_dic['data'].shape[1]), order='C'), dtype=n.single, requirements='C')
        self.data_dic['joints8'] = self.data_dic['joints8']/self.img_size;
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        alldata = [self.data_dic['data'], self.data_dic['joints8']]
        ind = self.data_dic['indmap']
        # Note that all image like data will use 'F' order
        # although it will require 'C' continuous data
        ndata = ind.shape[-1]
        self.data_dic['indmap'] = n.require(ind.reshape((-1, ndata), order='F'), dtype=n.single, requirements='C')
        alldata += [self.data_dic['indmap']  ]
        self.ind_dim = ind.shape[0] * ind.shape[1] * ind.shape[2]
        alldata += [n.ones((16, ndata),dtype=n.single, order='C')] 
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_size**2 * self.num_colors
        elif idx == 1:
            return 16
        elif idx == 2:
            # should be called after get_next_batch
            return self.ind_dim
        elif idx == 3:
            return 16 
    def get_plottable_data(self, data):
        return n.require(data + self.data_mean).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
    def get_joints(self):
        return self.data_dic['joints8'].reshape((8,2,-1), order='C')
    def get_num_parts(self):
        return 7
    
            
class LargeJoints8AndIndicatorFeatureAllDataProvider(DataProvider):
        def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
                DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
                self.data_mean = self.batch_meta['data_mean']
                self.num_colors = 3
                self.img_size = CONV_IMG_SIZE
                if 'indmap_para' not in self.batch_meta:
                    self.ind_dim = 7 * 8 * 8
                else:
                    self.ind_dim = self.batch_meta['indmap_para']['dim']
        def get_next_batch(self):
                load_time = time()
                self.data_dic = self.get_batch(self.curr_batchnum)
                ndata = self.data_dic['feature'].shape[1]
                self.data_dic['feature'] = n.require(self.data_dic['feature'], dtype=n.single, requirements='C')
                self.data_dic['joints8'] = n.require(self.data_dic['joints8'].reshape((16, ndata), order='C'), dtype=n.single, requirements='C')
                self.data_dic['joints8'] = self.data_dic['joints8']/self.img_size
                epoch, batchnum = self.curr_epoch, self.curr_batchnum
                self.advance_batch()
                alldata = [self.data_dic['feature'], self.data_dic['joints8']]
                ind = self.data_dic['indmap']
                # Note that all image like data will use 'F' order
                # although it will require 'C' continuous data
                # 0th body parts will start as idx 2
                ndata = ind.shape[-1]
                self.data_dic['indmap'] = n.require(ind.reshape((-1, ndata), order='F'), dtype=n.single, requirements='C')
                alldata += [self.data_dic['indmap']  ]
                self.ind_dim = 448 # next time I will write it into meta
                return epoch, batchnum, alldata
        def get_data_dims(self, idx=0):
                if idx == 0: # temp value, I will add it later in meta file
                        return 1600
                elif idx == 1:
                        return 16
                else:
                        # should be called after get_next_batch
                        return self.ind_dim
        def get_plottable_data(self, data):
            return None
            #return n.require(data).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
        def get_joints(self):
                return self.data_dic['joints8'].reshape((8,2,-1), order='C')
        def get_num_parts(self):
                return 7
    
class H36MMonoDataProvider(DataProvider):
        """
         This data provider can provide imgdata, mono_joints3d, joint_indicator map in order
        """
        def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
                DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
                self.data_mean = self.batch_meta['data_mean']
                self.num_colors = 3
                self.img_size = CONV_IMG_SIZE
                self.jtind_dim = self.batch_meta['ind_dim']['joint_indmap']
                self.normalize_mono_jt = 1200 
                # if it is set to be True, get_next_batch
                # get_next_batch will not load new dic  
                self.keep_dic = False
                self.njoints = self.batch_meta['njoints']
        def set_data_dic(self, dic):
                self.data_dic = dic
        def get_next_batch(self):
                if not self.keep_dic:
                    self.data_dic = self.get_batch(self.curr_batchnum)
                imgdata = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
                self.data_dic['mono_joints3d'] = n.require(self.data_dic['mono_joints3d'].reshape((self.njoints*3, self.data_dic['data'].shape[-1]), order='F'), dtype=n.single, requirements='C')
                self.data_dic['mono_joints3d'] = self.data_dic['mono_joints3d']/self.normalize_mono_jt
                alldata = [imgdata, self.data_dic['mono_joints3d'].copy()]
                ind = self.data_dic['joint_indmap']
                ndata = ind.shape[-1]
                self.data_dic['joint_indmap'] = n.require(ind.reshape((-1, ndata), order='F'), dtype=n.single, requirements='C')
                alldata += [self.data_dic['joint_indmap']  ]
                epoch, batchnum = self.curr_epoch, self.curr_batchnum
                self.advance_batch()
                return epoch, batchnum, alldata
        def get_data_dims(self, idx=0):
                if idx == 0:
                    return self.img_size**2 * self.num_colors
                elif idx == 1:
                    return self.njoints * 3
                else:
                    return self.jtind_dim[0] * self.jtind_dim[1] * self.njoints
        def get_plottable_data(self, data):
                return n.require(data + self.data_mean).reshape((self.img_size, self.img_size, self.num_colors, data.shape[1]), order='F')
        def get_joints(self):
                return self.data_dic['mono_joints3d'].reshape((self.njoints,3,-1), order='C')
        def get_num_parts(self):
                return 0
