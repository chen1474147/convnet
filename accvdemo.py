"""
Usage
python accvdemo.py -f MODELPATH --mode=accveval --images-folder=IMAGEFOLDER --mean-image-path=MEAN_IMAGE_PATH

Exmaple
python accvdemo.py -f ~/saved_models/Action14 --mode=accveval --images-folder=./testimages --mean-image-path=~/saved_models/ACCVModels/action14_meta.mat 

Note: --layer-def and --layer-params might need to be specified 

"""
import numpy as np
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *
import pylab as pl
import matplotlib.pyplot as plt
import Image
sys.path.append('./imodules')


import iutils as iu
import dhmlpe_utils as dutils
import dhmlpe_features 
import scipy.io as sio
import iread
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('/home/grads/sijinli2/Projects/DHMLPE/Python/src/')
sys.path.append('/media/M_FILE/cscluster/Projects/DHMLPE/Python/src/')

   
class TestConvNetError(Exception):
    pass
class TestConvNet(ConvNet):
    def init_data_providers(self):
        self.test_data_provider = FakeDataProvider()
        self.train_data_provider = FakeDataProvider()
        print 'I am Here <<<<<<<<<<<<<<<<<<<'
    def __init__(self, op, load_dict):
        ConvNet.__init__(self, op, load_dic)
        self.statistics = dict()
        self.temp_data = dict()
    def get_gpus(self):
        self.need_gpu = False
        if self.op.get_value('mode'):
            mode_value = self.op.get_value('mode')
            flag = mode_value in ['accveval']
            self.need_gpu |= flag
        if self.need_gpu:
            ConvNet.get_gpus( self )
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        self.train_data_provider = self.test_data_provider = Dummy()
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
    def init_model_state(self):
        pass
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)
    def load_images(self, images_path):
        def load_as_arr(p):
            img = Image.open(p)
            # For simplicity, I just resize the images
            # Please do the pre-processing, say do the cropping, according to your own setting
            return np.array(img.resize((112,112))).reshape((-1,1),order='F')
        return np.concatenate([load_as_arr(p) for p in images_path], axis=1)
    def do_accveval(self):
        images_folder = self.op.get_value('images_folder')
        # get all jpg file in images_folder
        allfiles = iu.getfilelist(images_folder, '.*\.jpg')
        images_path = [iu.fullfile(images_folder, p) for p in allfiles]
        n_image = len(images_path)
        images = self.load_images(images_path)
        mean_image_path = self.op.get_value('mean_image_path')
        mean_image = sio.loadmat(mean_image_path)['cropped_mean_image']
        mean_image_arr = mean_image.reshape((-1,1),order='F')
        input_images = images - mean_image_arr
        # pack input images into batch data
        data = [input_images, np.zeros((51,n_image),dtype=np.single),
                np.zeros((1700,n_image), dtype=np.single)]
        # allocate the buffer for prediction
        pred_buffer = np.zeros((n_image, 51),dtype=np.single)
        data.append(pred_buffer)

        ext_data = [np.require(elem,dtype=np.single, requirements='C') for elem in data]
        # run the model
        ## get the joint prediction layer indexes
        self.pred_layer_idx = self.get_layer_idx('fc_j2',check_type='fc')
        self.libmodel.startFeatureWriter(ext_data, self.pred_layer_idx)
        self.finish_batch()

        raw_pred = ext_data[-1].T
        pred = dhmlpe_features.convert_relskel2rel(raw_pred) * 1200.0

        # show the first prediction
        show_idx = 0
        img = np.array(Image.open(images_path[show_idx]))
        fig = pl.figure(0)
        ax1 = fig.add_subplot(121)
        ax1.imshow(img)
        ax2 = fig.add_subplot(122,projection='3d')
        cur_pred = pred[..., show_idx].reshape((3,-1),order='F')
        part_idx = iread.h36m_hmlpe.part_idx
        params =  {'elev':-94, 'azim':-86, 'linewidth':6, 'order':'z'}
        dutils.show_3d_skeleton(cur_pred.T, part_idx, params)
    def start(self):
        self.op.print_values()
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        if self.mode:
            if self.mode == 'accveval':
                self.do_accveval()
        plt.show()        
        sys.exit(0)
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'data_path', 'minibatch_size', 'layer_params', 'batch_size', 'test_only', 'test_one', 'shuffle_data', 'crop_one_border', 'external_meta_path'):
                op.delete_option(option)
        op.add_option('mode', 'mode', StringOptionParser, "The mode for evaluation")
        op.add_option('images-folder', 'images_folder', StringOptionParser, 'The folder for testing images')
        op.add_option('mean-image-path', 'mean_image_path', StringOptionParser, 'The path for mean image')
        op.options['load_file'].default = None
        return op
    

if __name__ == "__main__":

    nums = [3, 4, 5, 12, 14, 15]

    num = 14

    sys.argv = [None] * 6
    sys.argv[0] = 'accvdemo.py'
    sys.argv[1] = '-f'
    sys.argv[2] = '../accvmodel/Action%d' % num
    sys.argv[3] = '--mode=accveval'
    sys.argv[4] = '--images-folder=./testimages'
    sys.argv[5] = '--mean-image-path=../accvmodel/action%d_mean.mat' % num

    try:
        op = TestConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = TestConvNet(op, load_dic)
        model.start()
    except (UnpickleError, TestConvNetError, opt.GetoptError), e:
        print '-----------------'
        print "error"
        print e
        print '           '
