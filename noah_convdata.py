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

class NoahDataProviderError(Exception):
    pass

class CroppedImageNetDataProvider(CroppedImageDataProvider):
    """
    
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.labels = self.batch_meta['labels']                   #
        self.labelnames = self.batch_meta['labelnames']           #
        self.labelwords = self.batch_meta['labelwords']           #
        self.test = test

        ## Add alias name ( Since it will be called by cost function)
        self.batch_meta['label_names'] = self.batch_meta['labelnames']  #
        # override images_path
        self.images_path = map(lambda x:iu.fullfile(data_dir,x), \
                                map(lambda x:iu.getpath(x, 1),\
                                    self.batch_meta['images_path']))
        self.num_classes = 1000
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C'), \
                   np.require(np.asarray(self.data_dic['labels']).reshape((1,ndata)), dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    # _joint_batches use the parant class's method
    def get_batch(self, batch_num):
        """
        batch_num in self.image_range
        """
        dic = CroppedImageDataProvider.get_batch(self, batch_num)
        labels = map(lambda x:self.labels[0, x], dic['cur_batch_indexes'])
        dic['labels'] = labels;
        return dic
    def get_data_dims(self, idx=0):
        return iprod(self.input_image_dim) if idx==0 else 1

