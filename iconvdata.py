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
       
        # valid_idx = n.require(1 - n.max(n.isnan(self.data_dic['joints8']), axis=0), dtype=n.bool)
        # self.data_dic['data'] = n.require(self.data_dic['data'][...,valid_idx], requirements='C')        
        # self.data_dic['joints8'] = n.require(self.data_dic['joints8'][...,valid_idx],dtype=n.single, requirements='C')
        # self.data_dic['labels'] = n.require(self.data_dic['labels'][...,valid_idx], dtype=n.single, requirements='C')
        # normalize joints to [0,1]
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

