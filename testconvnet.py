import numpy
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

class TestConvNetError(Exception):
    pass
class TestConvNet(ConvNet):
    def __init__(self, op, load_dict):
        ConvNet.__init__(self, op, load_dic)
    def get_gpus(self):
        self.need_gpu = self.op.get_value('analyze_output') is not None
        self.need_gpu |= self.op.get_value('show_estimation') is not None
        self.need_gpu |= self.op.get_value('save_feature_name') is not None
        if self.need_gpu:
            ConvNet.get_gpus( self )
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
    def init_model_state(self):
        if self.op.get_value('analyze_output'):
            self.softmax_idx = self.get_layer_idx(self.op.get_value('analyze_output'),check_type='softmax')
        if self.op.get_value('show_estimation'):
            self.joint_idx = self.get_layer_idx(self.op.get_value('show_estimation'),check_type='fc')
        if self.op.get_value('save_feature_name'):
            if self.op.get_value('save_feature_path') is None:
                raise TestConvNetError(' Please Specify the path to save features')
            self.feature_idx = self.get_layer_idx(self.op.get_value('save_feature_name'))
        ### write feature here 
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)
    def plot_cost(self):
        raise TestConvNetError(' I haven''t finished this part yet')
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
        raise TestConvNetError(' I haven''t finished this part yet')
    def plot_filters(self):
        raise TestConvNetError(' I haven''t finished this part yet')
    def plot_prediction(self):
        raise TestConvNetError(' I haven''t finished this part yet')
    def do_write_features(self):
        raise TestConvNetError(' I haven''t finished this part yet')
    def show_prediction(self):
        data = self.get_next_batch(train=False)[2]
        num_classes = self.test_data_provider.get_num_classes()
        num_cases = data[0].shape[1]
        label_names = self.test_data_provider.batch_meta['label_names']
        preds = n.zeros((num_cases, num_classes), dtype=n.single)
        idx = self.op.get_value('label_idx')
        print 'label idx = ' + str( idx )
        data += [preds]
        
        # run the model
        self.libmodel.startFeatureWriter(data, self.softmax_idx)
        self.finish_batch()
        label_preds = preds.argmax(axis=1)
        err_idx = n.where(label_preds != data[idx][0,:])[0]
        print len(err_idx)
        print 'The total error is ' + str(len(err_idx)) + ' out of ' + str( num_cases) \
            + ':' + str( len(err_idx) * 100.0 / num_cases)
        confusion_matrix = n.zeros((num_cases, num_classes), dtype=n.int)
        for i in range( num_cases):
            t,p = data[idx][0,i], label_preds[i]
            confusion_matrix[t][p] += 1
        s = '    \t'
        for i in range( num_classes):
            s = s + label_names[i] + '\t'
        print s
        
        for i in range( num_classes):
            s = label_names[i] + '\t'
            for j in range(num_classes):
                s += str( confusion_matrix[i,j]) + '\t'
            print s
    def save_joints8_estimation(self, all_images, true_pose, est_pose, save_folder):
        import iconvnet_datacvt as icvt
        import Image
        import Stickmen
        from PIL import ImageDraw
        icvt.iu.ensure_dir( save_folder )
        num_data = all_images.shape[-1]
        for i in range( num_data ):
            img = all_images[..., i]
            img = Image.fromarray(img.astype('uint8'))
            draw = ImageDraw.Draw(img)
            tp = true_pose[...,i].reshape((8,2), order='C')
            ep = est_pose[...,i].reshape((8,2),order='C')
            # print tp
            # print ep
            Stickmen.draw_joints8(draw, tp, (255,0,0))
            Stickmen.draw_joints8(draw, ep, (0, 255, 0))
            img.save(icvt.iu.fullfile(save_folder, str(i) + '.jpg'))
    def save_feature(self):
        """
        Currently, only one layer can be saved
        """
        import iutils as iu
        import scipy.io as sio
        testdp = self.test_data_provider
        num_batches = len(testdp.batch_range)
        print 'There are ' + str(testdp.get_num_batches(self.data_path)) + 'in directory'
        print 'There are ' + str( num_batches ) + ' in range'
        iu.ensure_dir(self.save_feature_path)
        feature_name = self.op.get_value('save_feature_name')
        feature_dim = self.model_state['layers'][self.feature_idx]['outputs']
        for b in range(num_batches):
            epoch, b_num, data = self.get_next_batch(train=False)
            print '   Start writing batch......\t' + str(b_num)
            num_data = data[0].shape[-1]
            data += [n.zeros((num_data, feature_dim), dtype=n.single)]
            save_name = 'batch_feature_' + str(b_num) + '_' + feature_name 
            save_path = iu.fullfile(self.save_feature_path, save_name)
            self.libmodel.startFeatureWriter(data, self.feature_idx)
            self.finish_batch()
            d = dict()
            d['X'] = data[-1].transpose()
            d['batch_num'] = b_num
            d['Y'] = data[1]
            d['Y_other'] = data[2:-1] if len(data) > 3 else []
            print 'The len of data is ' + str(len(data))
            print 'The shape of X is' + str(d['X'].shape)
            print 'The shape of Y is' + str(d['Y'].shape)
            sio.savemat(save_path, d)
            pickle(save_path, d)

    def show_joints8_estimation(self):
        import iconvnet_datacvt as icvt
        data = self.get_next_batch(train=False)[2]
        num_data = data[0].shape[1]
        #### Testing code inside
        # res = n.zeros(( num_data, 16), dtype=n.single)
        # layer_idx = self.get_layer_idx('diff')
        # data += [res]
        # self.libmodel.startFeatureWriter(data, layer_idx)
        # self.finish_batch()
        # print data[1][:,0].transpose() * self.test_data_provider.img_size
        # print res[0,:] * self.test_data_provider.img_size
        # res = res ** 2
        # print res.sum(axis=0)
        # print res.sum(axis=0) / num_data
        
        # print self.joint_idx
        # self.joint_idx = self.get_layer_idx('fc_j2')
        # print self.joint_idx
        # return
        #####
        ests = n.zeros((num_data, 16), dtype=n.single)
        idx = self.joint_idx
        data += [ests]
        img_size = self.test_data_provider.img_size
        self.libmodel.startFeatureWriter(data, idx)
        self.finish_batch()
        
        ests = ests.transpose()
        sqdiff = ((data[1] - ests)**2).sum(axis=0)
        all_sqdiff = sqdiff.sum()
        print 'The normalized sqdiff is ' + str( all_sqdiff)
        print 'The normalized avg sqdiff is' + str( all_sqdiff / num_data)
        true_pose = data[1] * img_size
        est_pose = ests * img_size
        print len(data)
        #print true_pose[:,1].transpose()
        #print est_pose[:,1].transpose()
        e = icvt.calc_PCP_from_joints8( true_pose, est_pose, 0.5, True)
        e1 = icvt.calc_Classification_error_from_joints8(true_pose, est_pose, (img_size, img_size, 0))
        save_folder = op.get_value("save_images")
        if save_folder is not None:
            all_images = self.test_data_provider.get_plottable_data(data[0])
            self.save_joints8_estimation(all_images, true_pose, est_pose, save_folder)
            del all_images
        show_type = op.get_value("show_images")
        if show_type is None or len(true_pose) < 16:
            return
        num_row = num_col = 4
        show_num = num_row * num_col
        if show_type == 'furthest':
            idx = sorted( range(0, num_data), key = lambda x:sqdiff[x], reverse=True)[0:show_num]
        elif show_type == 'random':
            idx = r.sample( range(0,num_data), show_num)
        else:
            return
        from PIL import ImageDraw
        import Image
        import Stickmen
        data[0] = self.test_data_provider.get_plottable_data(data[0])
        for row in range(num_row):
            for col in range(num_col):
                pl_idx = row * num_col + col
                if (pl_idx >= len(true_pose)):
                    break;
                pl.subplot(num_row, num_col, pl_idx)
                img = data[0][...,idx[pl_idx]]
                img = Image.fromarray(img.astype('uint8'))
                draw = ImageDraw.Draw(img)
                tp = true_pose[..., idx[pl_idx]].reshape((8,2),order='C')
                ep = est_pose[..., idx[pl_idx]].reshape((8,2),order='C')
                s = 1
                Stickmen.draw_joints8(draw, tp, (255,0,0))
                Stickmen.draw_joints8(draw, ep, (0,255,0))
                # for i in range(len(tp)):
                #     draw.ellipse((tp[i,0] - s, tp[i,1]-s, tp[i,0] + s,tp[i,1] + s), fill=(255, 0, 0))
                #     draw.ellipse((ep[i,0] - s, ep[i,1]-s, ep[i,0] + s,ep[i,1] + s), fill=(0, 255,0))
                img = n.asarray(img)
                pl.imshow( img )
        
    

    def start(self):
        self.op.print_values()
        if self.analyze_output:
            self.show_prediction()
        if self.show_estimation:
            self.show_joints8_estimation()
        if self.save_feature_name:
            self.save_feature()
        pl.show()
        sys.exit(0)

    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'data_path'):
                op.delete_option(option)
        op.add_option("analyze-output", "analyze_output", StringOptionParser, "Show specified objective function", default="")
        op.add_option("label-idx", "label_idx", IntegerOptionParser, "The layer idx, with which the output compare") 
        op.add_option("show-estimation", "show_estimation", StringOptionParser, "Show pose estimation result")
        op.add_option("show-images", "show_images", StringOptionParser, "Whether to use the estimated images")
        op.add_option("save-images", "save_images", StringOptionParser, "Save the estimated images")
        op.add_option("save-feature-name", 'save_feature_name', StringOptionParser, "Save features in layers specified in save_features")
        op.add_option('save-feature-path', 'save_feature_path', StringOptionParser, "save layer feature in 'save_feature_path' ")
        op.options['load_file'].default = None
        
        return op

if __name__ == "__main__":
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
