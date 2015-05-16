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
import iutils as iu
import dhmlpe_utils as dutils
sys.path.append('/home/grads/sijinli2/Projects/DHMLPE/Python/src/')
sys.path.append('/media/M_FILE/cscluster/Projects/DHMLPE/Python/src/')
class TestConvNetError(Exception):
    pass
class TestConvNet(ConvNet):
    def __init__(self, op, load_dict):
        ConvNet.__init__(self, op, load_dic)
        self.statistics = dict()
        self.temp_data = dict()
    def get_gpus(self):
        self.need_gpu = self.op.get_value('analyze_output') is not None
        self.need_gpu |= self.op.get_value('show_estimation') is not None
        self.need_gpu |= self.op.get_value('save_feature_name') is not None
        self.need_gpu |= self.op.get_value('analyze_feature_name') is not None
        self.need_gpu |= self.op.get_value('test_only') is not None
        self.need_gpu |= self.op.get_value('do_evaluation') is not None
        if self.op.get_value('mode'):
            mode_value = self.op.get_value('mode')
            flag = mode_value in ['do_score_prediction', 'slp_server']
            self.need_gpu |= flag
        #self.need_gpu |= self.op.get_value('ubd_image_folder') is not None
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
            

        if self.op.get_value('analyze_feature_name'):
            if self.op.get_value('save_feature_path') is None:
                raise TestConvNetError(' Please Specify the path to save features')
            feature_name =  self.op.get_value('analyze_feature_name')
            self.feature_idx = self.get_layer_idx(feature_name)
            # layer_list = [(9,1),(3,2),(5,1),(3,2), (5,1),(3,2)]
            # d = {'conv1':0, 'pool1':1, 'conv2':2,'pool2':3,'conv3':4,\
            #      'pool3':5, 'conv3_0':4, 'conv3_1':4, 'conv3_max':4}
            # extra_d = {'fc_i2':-1, 'fc_ij2':-2, \
            #            'fc_ubd2':-3, 'cnorm1':-4, \
            #            'data':-5, 'rnorm2':-5, 'rnorm1':-6}
            # if feature_name not in d and feature_name not in extra_d:
            #     raise TestConvNetError('feature %s is not supported' % feature_name)
            # if feature_name in d:
            #     self.layer_filter_size_list = layer_list[0:d[feature_name]+1]
            l = self.get_backtrack_layer_list(self.feature_idx)
            #self.layer_filter_size_list = self.get_backtrack_filter_size_list(l[:-1])
            # Should include the size of this layer !!!!!
            self.layer_filter_size_list = self.get_backtrack_filter_size_list(l)
            ### write feature here
    def get_backtrack_filter_size_list(self,l):
        res_l = []
        convset = set(['conv', 'pool'])
        for x in l:
            if x['type'] == 'conv':
                res_l += [(x['filterSize'][0], x['stride'][0], -x['padding'][0])]
            elif x['type'] == 'pool':
                res_l += [(x['sizeX'],x['stride'], x['start'])]
            elif x['type'] == 'rnorm':
                continue
                #res_l += [(x['size'], 1, -x['size']/2)]
            else:
                continue
        print res_l
        return res_l
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)
    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0][0]:
            raise TestConvNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
        train_errors = [o[0][self.show_cost][self.cost_idx] for o in self.train_outputs]
        test_errors = [o[0][self.show_cost][self.cost_idx] for o in self.test_outputs]
        numbatches = len(self.train_batch_range)
        test_errors = np.row_stack(test_errors)
        test_errors = np.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]
        if self.batch_size == -1:
            numepochs = len(train_errors) / int(numbatches)
        else:
            numepochs = len(train_errors) * self.batch_size  / int(len(self.train_batch_range))
        pl.figure(1)
        x = range(0, len(train_errors))

        print 'numepochs=%d' % numepochs
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        print test_errors[-10:]
        log_scale = False
        if log_scale:
            pass
        else: 
            pl.ylim([0, np.median(test_errors[-len(test_errors)/10:-1])*2])
            # print np.median(test_errors[-len(test_errors)/10:-1])*2, ',,,,,,,,'
        
        pl.legend()
        if self.batch_size == -1:
            ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        else:
            t = np.ceil(numbatches / self.batch_size)
            ## approximate the time for change 
            ticklocs = range(int(t), len(train_errors), int(numbatches/self.batch_size))
        epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
        if self.batch_size == -1:
            ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))
        else:
            t = np.ceil( numbatches / self.batch_size) * epoch_label_gran
            ticklabels = map(lambda x: str(x[1] * self.batch_size/numbatches) if np.floor(x[1] * self.batch_size/numbatches) % epoch_label_gran == 0  else '', enumerate(ticklocs))
        # pl.gca().set_yscale('log')
        pl.plot(x, test_errors, 'r-', label='Test set')
        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')

        pl.ylabel(self.show_cost)
        pl.title(self.show_cost)
        
        #raise TestConvNetError(' I haven''t finished this part yet')
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
        import Image
        import Stickmen
        from PIL import ImageDraw
        iu.ensure_dir( save_folder )
        num_data = all_images.shape[-1]
        for i in range( num_data ):
            img = all_images[..., i]
            img = Image.fromarray(img.astype('uint8'))
            draw = ImageDraw.Draw(img)
            tp = true_pose[...,i].reshape((8,2), order='C')
            ep = est_pose[...,i].reshape((8,2),order='C')
            # print tp
            # print ep
            Stickmen.draw_joints8(draw, tp, (255,0,0), dual_color=False,width=2)
            Stickmen.draw_joints8(draw, ep, (0, 255, 0), dual_color=False,width=2)
            img.save(iu.fullfile(save_folder, str(i) + '.png'))
    def save_feature(self):
        """
        Currently, only one layer can be saved
        This function is designed for writing features for pose data
        
        """
        import scipy.io as sio
        testdp = self.test_data_provider
        num_batches = len(testdp.batch_range)
        print 'There are ' + str(testdp.get_num_batches(self.data_path)) + ' in directory'
        if self.test_data_provider.batch_size > 0:
            num_batches = (num_batches - 1)/ self.test_data_provider.batch_size + 1
        if self.test_one:
            num_batches = min(num_batches, 1)
        print 'There are ' + str( num_batches ) + ' in range'
        iu.ensure_dir(self.save_feature_path)
        feature_name = self.op.get_value('save_feature_name')
        feature_dim = self.model_state['layers'][self.feature_idx]['outputs']
        print 'Feature dim is %d' % feature_dim
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
            cur_batch_indexes = self.test_data_provider.data_dic['cur_batch_indexes']
            # d['Y_other'] = data[2:-1] if len(data) > 3 else []
            ####### WARN BEGIN ################
            # for human eva fake experiments
            # d['images_path'] = [self.test_data_provider.images_path[x] for x in cur_batch_indexes]
            # d['Y'] = np.concatenate(map(lambda x:self.test_data_provider.batch_meta['RelativeSkel_Y3d_mono_body_backup'][...,x].reshape((-1,1),order='F'), cur_batch_indexes),axis=1)
            print d['Y'].shape
            d['cur_batch_indexes'] = cur_batch_indexes
            ####### WARN END ################
            print 'The len of data is ' + str(len(data))
            print 'The shape of X is' + str(d['X'].shape)
            print 'The shape of Y is' + str(d['Y'].shape)
            ##sio.savemat(save_path, d)
            pickle(save_path, d)
    @classmethod
    def parse_params(cls, s):
        l = s.split(',')
        res_l = []
        for x in l:
            if x.find('@') != -1:
                a = x.split('@')
                res_l += [(a[0], int(a[1]))]
            else:
                res_l += [x]              
        return res_l
    def analyze_feature(self):
        # analyze feature
        import scipy.io as sio
        testdp = self.test_data_provider
        if (not 'batch_size' in testdp.__dict__.keys()) or testdp.batch_size < 0:
            num_batches = len(testdp.batch_range)
        else:
            num_batches = (len(testdp.batch_range) -1) / int(testdp.batch_size) + 1
        print 'There are ' + str(testdp.get_num_batches(self.data_path)) + ' in directory'
        print 'There are ' + str( num_batches ) + ' in range'
        iu.ensure_dir(self.save_feature_path)
        feature_name = self.op.get_value('analyze_feature_name')
        feature_dim = self.model_state['layers'][self.feature_idx]['outputs']
        feature_channel = self.op.get_value('feature_channel')
        abs_sum_feature = n.zeros([feature_dim, 1], dtype=n.float32)
        print 'Feature dimension = ' + str(feature_dim)
        tot_data = 0
        #np.random.seed(7)
        if self.test_one:
            num_batches = 1
        if self.calc_hist:
            self.feature_params = self.parse_params(self.calc_hist)
        if self.calc_hist and self.feature_params[0] == 'actjoint':
            import indicatormap
            ## Assume the 0-layer will be RGB data
            l = self.get_backtrack_layer_list(self.feature_idx)
            layer_filter_size_list = self.get_backtrack_filter_size_list(l) 
            fs = iu.get_conv_fs(layer_filter_size_list)
            s = np.int(np.sqrt(self.model_state['layers'][0]['outputs']/3.0))
            ind_map_instance = indicatormap.IndicatorMap((s,s,3), \
                                                fs[0],
                                                fs[1],
                                                True)
        for b in range(num_batches):
            epoch, b_num, data = self.get_next_batch(train=False)
            print '   Start writing batch......\t' + str(b_num)
            num_data = data[0].shape[-1]
            data += [n.zeros((num_data, feature_dim), dtype=n.single)]
            self.libmodel.startFeatureWriter(data, self.feature_idx)
            self.finish_batch()
            abs_sum_feature += np.abs(data[-1]).sum(axis=0).reshape((-1,1))
            tot_data += num_data
                        
            if self.show_response == 'random': 
                num_to_display = min(16, num_data)  
                perm = np.random.randint(0, num_data, num_to_display)
                plot_data  = self.test_data_provider.get_plottable_data(data[0])[..., perm]/255.0
                s = np.sqrt(feature_dim / feature_channel)
                plot_response = data[-1].transpose().reshape((s,s,feature_channel,num_data),order = 'F')[...,perm]
                self.display_image_response(plot_data, plot_response)
            elif self.save_response ==  'separate':
                
                plot_data = self.test_data_provider.get_plottable_data(data[0])/255.0
                s = np.sqrt(feature_dim / feature_channel)
                plot_response = data[-1].transpose().reshape((s,s,feature_channel, num_data), order='F')
                self.save_image_response(plot_data, plot_response, \
                                        'batch_' + str(b_num) + \
                                        '_feature_' + feature_name)
            elif self.save_res_patch in set(['all','average', 'allpatchdata', 'allpatchdata-feature']):
                plot_data = self.test_data_provider.get_plottable_data(data[0])/255.0
                s = np.sqrt(feature_dim / feature_channel)
                plot_response = data[-1].transpose().reshape((s,s,feature_channel, num_data), order='F')
                print plot_data.shape, plot_response.shape
                self.save_image_res_patch(plot_data, plot_response, \
                                          'batch_' + str(b_num) + \
                                          '_feature_%s' % feature_name)
            elif self.calc_hist and self.feature_params[0] in set(['actjoint']):
                dp = self.test_data_provider
                options = dict()
                options['num_joints'] = dp.num_joints
                options['abs'] = True
                options['add_background'] = True
                options['normalize'] = 'max' #'sum'
                if len(self.feature_params) > 1 and self.feature_params[1][0]=='occ':
                    print 'Using Occlusion information'
                    options['joint_occ'] = dp.data_dic['occ_body']
                ndata = data[0].shape[-1]              
                num_joints = options['num_joints']
                jtname='joints_2d'
                indmap = ind_map_instance.get_joints_indicatormap(dp.data_dic[jtname].reshape((2,-1),order='F').T ).reshape((-1,num_joints, ndata),order='F')
                self.statistics['options'] = options
                # self.statistics['indmap'] = indmap
                # self.statistics['joints_2d'] = dp.data_dic[jtname]
                # self.statistics['data'] = self.test_data_provider.get_plottable_data(data[0])/255.0
                self.statistics['activation'] = data[-1].transpose().reshape((-1, feature_channel, ndata),order='F').copy()

                self.calc_activation_joint_histogram(indmap, \
                                data[-1].transpose().reshape((-1, feature_channel, ndata),order='F'),options)
            elif self.save_indmap_show == 'all':
                plot_data = self.test_data_provider.get_plottable_data(data[0])/255.0
                
                ndata = plot_data.shape[-1]
                s = np.sqrt(data[-1].shape[-1]/feature_channel)
                ind_maps = data[-1].transpose().reshape((feature_channel,s,s,ndata), order='F')

                self.save_image_indmap(plot_data, ind_maps, \
                                           'batch_' + str(b_num) + '_indmap') 
                
        abs_sum_feature /= (tot_data+ 0.0)
        d = dict()
        save_name = 'batch_analysis_' + feature_name 
        save_path = iu.fullfile(self.save_feature_path, save_name)
        d['abs_sum'] = abs_sum_feature
        pickle(save_path, d)
        self.display_feature(np.abs(abs_sum_feature), feature_channel, isrgb=False)
    def makeindmap(self,indmap,s,backgroud = None):
        """
        Create indmap so that the size of indmap matches the original image
        """
        if s == 1:
            raise TestConvNetError('s should be larger than 1')
        wr,wc = indmap.shape[0],indmap.shape[1]
        stride = 12
        filter_size = 30
        if backgroud is None:
            res = np.zeros((wr*s,wc*s,3),dtype=np.float32)
        else:
            alpha = 0.6
            if backgroud.shape != (wr*s,wc*s,3):
                raise TestConvNetError('Error size of backgroud')
            res = alpha * backgroud.copy()
        for i in range(wr):
            for j in range(wc):
                res[i*stride:i*stride+filter_size,j*stride:j*stride+filter_size]+= (1-alpha)* indmap[i,j]
        m = np.minimum(res,1)
        
        return m
    def makeindmapKDE(self,indmap,s, background):
        """
        create detection map using kernel desity estimation

        Rely on  filter_size,
                 stride
        """
        import ipyml
        from ipyml.probability import pfunc
        sp = background.shape
        res = np.zeros((sp[0], sp[1]),dtype=np.float32)
        wr,wc = indmap.shape[0], indmap.shape[1]
        filter_size = 30
        stride = 12
        cov = np.asarray([[(2.0/filter_size)**2,0],[0,(2.0/filter_size)**2]])
        if 'g' in self.temp_data:
            g = self.temp_data['g']
        else:
            g = pfunc.Gaussian2D((sp[0],sp[1]),cov=cov,invcov=False)
            self.temp_data['g'] = g
        center_r = sp[0]
        center_c = sp[1]
        g = g/g.max()
        for r in range(wr):
            for c in range(wc):
                # calcuate the center of detection window
                rr = (r * stride + r * stride + filter_size-1)/2
                cc = (c * stride + c * stride + filter_size-1)/2
                offset_r = center_r - rr
                offset_c = center_c - cc
                res = res + g[offset_r:offset_r+sp[0],offset_c:offset_c+sp[1]] * indmap[r,c]
        idx = np.argmax(res)
        res = np.tile(res.reshape((res.shape[0],res.shape[1],1)),[1,1,3])
        mr = idx / sp[1]
        mc = idx - mr * sp[1]
        hf = filter_size/2
        box = np.asarray([mc -hf,mr -hf,mc + hf, mr + hf])
        return res/3, box


    def save_image_indmap(self, imgdata, indmap, prename):
        if self.save_feature_path is None:
            raise TestConvNetError('Please specify save-feature-path')
        ndata =  imgdata.shape[-1]
        if self.matched_path is not None:
            matched = unpickle(self.matched_path)
            print 'I will removed % d image' % (ndata - matched.size)  
        else:
            matched = range(ndata)
        import Image
        from PIL import ImageDraw
        iu.ensure_dir( self.save_feature_path)
        print 'Begin to save indmap There are %d in total' % ndata
        sp = imgdata.shape
        if sp[0] != sp[1]:
            raise TestConvNetError('Only square images are supported')
        m = 3 # margin size
        num_parts = indmap.shape[0]
        bigimage = np.zeros((sp[0],sp[0] + (sp[0] + m) * indmap.shape[0], sp[2]),dtype=np.float)
        s = sp[0]/indmap.shape[1]
        logistic = lambda x: 1.0/(1 + np.exp(-x))
        indmap = logistic(indmap)
        dp = self.test_data_provider
        if self.ind_type == 'joint_ind':
            tpose = self.transform_pose_back_to_oriimg_coor(dp, dp.data_dic['joints8'] * dp.img_size, False)
            if len(matched) != tpose.shape[-1]:
                raise TestConvNetError('Sample number of groudth-truth and estimation doesn''t match')
            epose = np.zeros((num_parts, 2, tpose.shape[-1]))
        for i in matched:
            if i < 0:
                continue
            bigimage[:] = 0
            bigimage[:sp[0],:sp[1],:] = imgdata[...,i]
            bbox = np.zeros((4,num_parts),dtype=np.float32)
            for j in range(num_parts):
                # draw margin
                c_start = sp[0] + j * (sp[0] + m)
                bigimage[:sp[1],c_start:c_start+m,1] = 1.0
                bigimage[:sp[1],c_start:c_start+m,2] = 1.0
                t1 = c_start + m
                t2 = c_start + m + sp[0]
                indshow, box = self.makeindmapKDE(indmap[j,:,:,i],s,imgdata[...,i])
                bbox[...,j] = box
                # import imgproc
                # bigimage[:sp[0],t1:t2,:] = imgproc.imgeq(np.minimum(1,indshow/2.0))/255.0
                bigimage[:sp[0],t1:t2,:] = imgdata[...,i]
            saveimg = Image.fromarray(np.require(bigimage * 255, dtype=np.uint8))
            draw = ImageDraw.Draw(saveimg)
            inc = [ 0 if x == 0 else 0 for x in range(num_parts)]
            # clist = iu.getrgbtable(num_parts)
            if self.ind_type == 'joint_ind':
                offset = [(0,0) if x == 0 else (0,0) for x in range(num_parts)]
                clist = [(255,255,255), (0,0, 255), (255,255,0), (255,0,0),\
                         (72,118,255), (255,0,255), (0,191,255),(224,102,255)]
            else:
                offset = [(0,-10) if x == 0 else (0,0) for x in range(num_parts)]
                clist = [(255,255,255), (0,0, 255), (255,0,0),\
                         (72,118,255),  (0,191,255),(255,0,255),(224,102,255)] 	
            for j in range(num_parts):
                c_start = sp[0] + j * (sp[0] + m)
                t1 = c_start + m + offset[j][0]
                t2 = offset[j][1]
                draw.rectangle((bbox[0,j]+t1-inc[j],bbox[1,j]+t2-inc[j],bbox[2,j]+t1+inc[j],bbox[3,j]+t2+inc[j]),outline=clist[j])
                # draw all on the left most image
                t1 = t1 - c_start - m
                if self.ind_type != 'joint_ind':
                    draw.rectangle((bbox[0,j]+t1-inc[j],bbox[1,j]+t2-inc[j],bbox[2,j]+t1+inc[j],bbox[3,j]+t2+inc[j]),outline=clist[j])
                else:
                    cc = ((bbox[0,j] + bbox[2,j])/2,(bbox[1,j] + bbox[3,j])/2)
                    draw.rectangle((cc[0]+t1-3,cc[1]+t2-3,cc[0]+t1+3,cc[1]+t2+3),fill=clist[j])
                    ## epose has the same dimension as i
                    epose[j,:,i] = [cc[0],cc[1]] # 
            savename = prename + '_%d.png' % i 
            saveimg.save(iu.fullfile(self.save_feature_path, savename))
        print self.ind_type
        if self.ind_type == 'joint_ind':
            import  iconvnet_datacvt as icvt
            ## The same as in show_joints8_estimation
            epose = self.transform_pose_back_to_oriimg_coor(dp, epose.reshape((-1, tpose.shape[-1]),order='C'), False)
            e = icvt.calc_PCP_from_joints8( tpose, epose, 0.5, True)

    def calc_activation_joint_histogram(self, joint_indicator_map, activations, options):
        import dhmlpe_analysis as da
        res_l = da.calc_activation_joint_histogram(joint_indicator_map, \
                                                    activations, \
                                                    options)
        for name in res_l:
            if not name in self.statistics:
                self.statistics[name] = res_l[name]
            else:
                self.statistics[name] += res_l[name]
            print self.statistics[name][:,0].T
        pickle(iu.fullfile(self.save_feature_path, 'hist_statistics'), self.statistics)
    def save_image_res_patch(self,imgdata,resdata, prename, reorder_channel=True):
        import scipy.io as sio
        if self.save_feature_path is None:
                raise TestConvNetError('Please specipy save-feature-path ')
        ndata = imgdata.shape[-1]
        print 'Begin to save reponse, there are ' + str(ndata) + ' in total'
        #prename = 'net10_24_' + prename
        layer_list = self.layer_filter_size_list
        print '======'
        print 'Using Layer structure ' + str(layer_list)
        print '=======' 
        sp = resdata.shape
        resdata = resdata.reshape((sp[0]*sp[1],sp[2],sp[3]),order='F')
        am = np.argmax(resdata, axis=0)
        MAX_IMAGE_ROW=8
        nrow = (sp[2]-1)/MAX_IMAGE_ROW + 1 + 1
        t = 0
        plt.rcParams['figure.figsize'] = 15, min(nrow * 2, 15)
        box = list(iu.back_track_filter_range(layer_list, (0,0,0,0)))
        print 'back_track result is ',
        print  box
        avgpatches = np.zeros((box[2]-box[0]+1, box[3]-box[1]+1 ,imgdata.shape[2], sp[2]), dtype=np.float32)
        avgc = np.zeros((sp[2]), dtype=np.int32)                
        tc = imgdata[...,0].shape[2]
        curpatches = np.zeros((box[2]-box[0] +1,box[3]-box[1]+1, tc, sp[2]),dtype=np.float32)
        sumpatches = np.zeros( imgdata[...,0].shape )
        indpatches = np.zeros( imgdata[...,0].shape,dtype=np.bool )
        patches = np.zeros( imgdata[...,0].shape )
        clist = range(sp[2])
        saveimgall = True if self.save_res_patch =='all' else False
        savepatchall = True if self.save_res_patch in ['allpatchdata', 'allpatchdata-feature'] else False
        savepatch_feature_all = True if self.save_res_patch == 'allpatchdata-feature' else False
        if savepatchall:
            allpatches = np.zeros((box[2]-box[0]+1,box[3]-box[1]+1, tc, sp[2], sp[3]),dtype=np.float32)
        if savepatch_feature_all:
            # The feature for patch at k-th channel, n-th data is allfeatures[:,k,n]
            allfeatures = np.zeros((sp[2], sp[2], sp[3]), dtype=np.float32)
        if reorder_channel is True and sp[2] == 16:
            if  self.op.get_value('analyze_feature_name') == 'conv3': 
                clist = [ 9, 2, 8,12,4, 7, 0, 3,\
                        15, 6,14, 5,1,11,10,13]
            else: # conv2
                clist = [15, 2, 0, 5, 8,11, 1, 7,\
                         10,13, 4,12, 6,14, 9, 3]
        if reorder_channel is True and sp[2] == 32:
            if  self.op.get_value('analyze_feature_name') == 'conv3': 
                clist = [22, 23, 6, 7, 29, 8, 19, 16, \
                         30, 25, 21, 27, 9, 11, 3, 24,\
                         2,  4,  10, 12, 20, 0, 1, 14, \
                         5, 13, 15,  21, 17, 26,14,28]
                clist = [0, 11, 24, 5, 22, 3, 13, 16, 31,\
                          18, 14, 21, 12, 26, 8, 9, 25, 17,\
                           6, 27, 20, 29, 19, 4, 10, 15, 2,
                            23, 30, 1, 28, 7] # for network 26, c8k19
            elif  self.op.get_value('analyze_feature_name') == 'conv2': 
                clist = [15, 8, 0, 27, 22, 9, 19, \
                         1, 13, 25, 2, 31, 29, 4, \
                         21, 3, 24, 17, 20, 27, 18,\
                          6, 10, 5, 11, 30, 7, 16, \
                          28, 23, 12, 14] # for network 26
        # The dimension in sp is nrow, ncol, nchannel, ndata
        for i in range(sp[3]):
            sumpatches[:] = 0
            indpatches[:] = False
            curpatches[:] = 0
            for cidx in range(sp[2]):
                channel = clist[cidx]
                #patches[:] = 0
                index = am[channel, i]
                if savepatch_feature_all:
                    allfeatures[:,channel,i] = resdata[index,:,i]
                c = index / sp[0]
                r = index - c * sp[0]
                bbox = list(iu.back_track_filter_range(layer_list, (r,c,r,c)))
                # print bbox,
                # print ' cidx=%d ' % cidx
                if abs(bbox[0]) >= imgdata.shape[0] or \
                  abs(bbox[1]) >= imgdata.shape[1]:
                    continue
                # bbox[2] = min(bbox[2], imgdata.shape[0])
                # bbox[3] = min(bbox[3], imgdata.shape[1])
                ### copy with padding if neccessary
                s_r = max(bbox[0], 0)
                s_c = max(bbox[1], 0)
                t_r = 0 if bbox[0] >=0  else -bbox[0]
                t_c = 0 if bbox[1] >=0  else -bbox[1] 
                l_r = min(bbox[2] - bbox[0] + 1 - t_r, imgdata.shape[0] - s_r)
                l_c = min(bbox[3] - bbox[1] + 1 - t_c, imgdata.shape[1] - s_c)
                if bbox[0] < 0:
                    print 's_r = %d, s_c = %d, t_r = %d, t_c = %d, l_r = %d, l_c = %d' % (s_r, s_c, t_r, t_c, l_r, l_c)
                curpatches[t_r:t_r+l_r,t_c:t_c+l_c,:,cidx] = imgdata[s_r:s_r + l_r, s_c:s_c + l_c,:, i]                 
                ###
                avgpatches[:bbox[2]-bbox[0]+1,:bbox[3]-bbox[1]+1,:,cidx] += \
					curpatches[:,:,:,cidx]
                avgc[cidx] += 1
                #patches[:bbox[2]+1, bbox[1]:bbox[3]+1,:] = curpatches[:,:,:,cidx]
                sumpatches[s_r:s_r + l_r, s_c:s_c+l_c,:] += imgdata[s_r:s_r + l_r, s_c:s_c + l_c,:, i]
                indpatches[s_r:s_r+l_r, s_c:s_c+l_c,:] = True
                
            coverpatch = np.zeros( imgdata[...,0].shape)
            coverpatch[indpatches] = imgdata[indpatches,i]
            if saveimgall:
                self.Show_img_and_patches(imgdata[...,i], curpatches)
                plt.savefig(iu.fullfile(self.save_feature_path, 'img_' + str(i) + '_backtrack.png'),bbox_inches='tight')
                #plt.savefig(iu.fullfile(self.save_feature_path, prename + '_' + str(i) + '.png'))
            if savepatchall:
                allpatches[...,i] = curpatches
            print '%d saved' % i
        if savepatchall:
            if allpatches.shape[-2] <= 16:
                pickle(iu.fullfile(self.save_feature_path, 'patch_data'), allpatches)
            else:
                for i in range(allpatches.shape[-2]):
                    pickle(iu.fullfile(self.save_feature_path, 'patch_data_map_%d' % i), allpatches[:,:,:,i,:])
                    sio.savemat(iu.fullfile(self.save_feature_path, 'patch_data_map_%d' % i), {'data':allpatches[:,:,:,i,:]})
        if savepatch_feature_all:
            for i in range(sp[2]):
                    pickle(iu.fullfile(self.save_feature_path, 'patch_feature_map_%d' % i), allfeatures[:,i,:])
                    sio.savemat(iu.fullfile(self.save_feature_path, 'patch_feature_map_%d' % i), {'feature':allfeatures[:,i,:]})
        self.Show_multi_channel_image(avgpatches, True, avgc)
        plt.savefig(iu.fullfile(self.save_feature_path, prename + '_all_filter_avg_imgeq.png'))
        # self.Show_multi_channel_image(avgpatches, False, avgc)
        # plt.savefig(iu.fullfile(self.save_feature_path, prename + '_all_filter_avg_ori.png'))
        if 'avgpatches' not in self.statistics:
            self.statistics['avgpatches'] = avgpatches
            self.statistics['avgc'] = avgc
        else:
            self.statistics['avgpatches'] += avgpatches
            self.statistics['avgc'] += avgc
        self.Show_multi_channel_image(self.statistics['avgpatches'], 'imgeq', self.statistics['avgc'])
        plt.savefig(iu.fullfile(self.save_feature_path, '_all_to_' + prename + '_all_filter_avg_imgeq.png'), bbox_inches='tight')
        self.Show_multi_channel_image(self.statistics['avgpatches'], 'maptorange', self.statistics['avgc'])
        plt.savefig(iu.fullfile(self.save_feature_path, '_all_to_' + prename + '_all_filter_avg_maptorange.png'), bbox_inches='tight')
        
        pickle(iu.fullfile(self.save_feature_path, 'statistics'), self.statistics) 
    def Show_img_and_patches(self, imgdata, patches):
        """
        """
        
        MAX_IMAGE_ROW = 8
        sp = imgdata.shape
        sp1 = patches.shape
        num_patch = patches.shape[-1]
        nrow = (num_patch - 1)/MAX_IMAGE_ROW + 1
        fr = plt.gca()
        fr.axes.get_xaxis().set_visible(False)
        fr.axes.get_yaxis().set_visible(False)
        mg = 5 # margin
        height = max(imgdata.shape[0], nrow * patches.shape[0])
        width = imgdata.shape[1] + MAX_IMAGE_ROW *(patches.shape[1] + mg)
        
        bigimage = np.zeros((height, width, imgdata.shape[2]),dtype=np.float32)
        ## put imgdata
        
        bigimage[:sp[0],:sp[1],:] = imgdata
        ## put patches data
        per_h = height / nrow
        cur_fig = plt.gcf()
        cur_fig.set_size_inches(min(16,MAX_IMAGE_ROW),min(nrow*1.6, 10))
        for i in range(num_patch):
            r = i / MAX_IMAGE_ROW
            c = i - (r * MAX_IMAGE_ROW)
            dr = r * per_h
            dc = sp[1] + (sp1[1] + mg) * c 
            bigimage[dr:sp1[0] + dr,dc:sp1[1] + dc,:] = patches[...,i]
        plt.imshow(bigimage)
        
    def Show_multi_channel_image(self, allimages, enhance=False, num =None):
        """
        This function will show multi channel image in one draw
        num is the number of images in each channel
        """
        import imgproc
        MAX_IMAGE_ROW = 8
        nchannel = allimages.shape[-1]
        nrow = (nchannel - 1)/MAX_IMAGE_ROW + 1
        cur_fig = plt.gcf()
        cur_fig.set_size_inches(min(16,MAX_IMAGE_ROW*1.6),min(nrow*1.6, 10))
        if enhance == 'imgeq':
            f = lambda(x):imgproc.imgeq(x)/255.0
        elif enhance == 'maptorange':
            f = lambda(x):imgproc.maptorange(x,[0,1])
        else:
            f = lambda x: x
        for channel in range(nchannel):
            plt.subplot(nrow, MAX_IMAGE_ROW, channel + 1)
            fr1 = plt.gca()
            fr1.axes.get_xaxis().set_visible(False)
            fr1.axes.get_yaxis().set_visible(False)
            plt.title('%d' % (channel + 1))
            if num is not None:
                plt.imshow( f(allimages[...,channel]/num[channel])) 
            else:
                plt.imshow( f(allimages[...,channel])) 
        
    def save_image_response(self, imgdata, resdata, prename):
        if self.save_feature_path is None:
            raise TestConvNetError('Please specipy save-feature-path ')
        ndata = imgdata.shape[-1]
        print 'Begin to save reponse, there are ' + str(ndata) + ' in total'
        MAX_IMAGE_ROW = 8

        import imgproc
        n_res = resdata[...,0].shape[-1]
        nrow = (n_res - 1) / MAX_IMAGE_ROW + 1 + 1
        #plt.rcParams['figure.figsize'] = 15, max(10, nrow*1.3) # width, height

        for i in range(ndata):
            print '%d' % i
            img = imgdata[...,i]
            res = resdata[...,i]
            # plt.subplot(nrow, MAX_IMAGE_ROW, 1)
            # plt.imshow(img)
            sp = (res.shape[0], res.shape[1])
            imgproc.turn_off_axis()
            bigimage = imgproc.BigImagePlot([70,70], (nrow, MAX_IMAGE_ROW), 3,3,(1,1,1))
            bigimage.set_same_size(False)
            bigimage.subplot(img,0,0)

            for j in range(1, nrow):
                for k in range(MAX_IMAGE_ROW):
                    idx = j * MAX_IMAGE_ROW + k
                    # plt.subplot(nrow, MAX_IMAGE_ROW, idx + 1)
                    # plt.imshow(res[..., idx - MAX_IMAGE_ROW].reshape(sp))
                    # imgproc.turn_off_axis()
                    tmp = res[..., idx - MAX_IMAGE_ROW].reshape(sp) / max(np.abs(res[...,idx-MAX_IMAGE_ROW].flatten().max()), 1e-9)                    
                    bigimage.subplot(tmp, j,k)
            savename = iu.fullfile(self.save_feature_path, prename + '_' + str(i) + '.png')
            #plt.savefig(savename)
            bigimage.save(savename)
        
    def prepare_feature_imgae(self, fimg):
        import imgproc
        channel = fimg.shape[-1]
        if channel == 3:
            return fimg
        else:
            return imgproc.imgeq(np.abs(fimg).sum(axis=-1))
    def display_image_response(self, images, responses):
        # image will be in ... x num_data format
        # responses will be ... x num_data format
        MAX_IMG_ROW = 4
        MAX_ROW = 4
        ndata = min(images.shape[-1], MAX_IMG_ROW * MAX_ROW)
        nrow = (ndata-1)/MAX_IMG_ROW + 1
        pl.subplots(2,2)
        import matplotlib.cm as cm
        for i in range(ndata):
            pl.subplot(nrow, MAX_IMG_ROW*2, i*2 + 1) 
            #pl.subplot(2, 2, 0)
            cur_image = images[..., i]
            cur_resp = responses[..., i]
            pl.imshow(cur_image)
            pl.subplot(nrow, MAX_IMG_ROW * 2, (i * 2) + 2)
            cur_resp =self.prepare_feature_imgae(cur_resp)/255.0 
            
            pl.imshow(cur_resp)
            #pl.imshow(cur_resp, cmap=cm.RdBu_r)
        plt.show()
        
    def display_feature(self, imgdata, channel, isrgb = True):
        import imgproc
        if (imgdata.size % channel) != 0:
            raise TestConvNetError('size of image %d can not divide number of channel %d' % (imgdata.size , channel))
        
        s = np.sqrt(imgdata.size / channel)
        if channel == 3:
            imgdata = imgdata.reshape(s,s,channel, order='F')/255.0
        else:
            #imgdata = (imgdata.reshape(s,s,channel, order='F')).reshape((s,s))
            imgdata = imgdata.reshape(s,s,channel, order='F')
        import matplotlib.cm as cm
        #imgdata = iu.imgproc.imgeq(imgdata)/255.0
        # plt.hist(imgdata.flatten(), 1000)
        # plt.show()
        # return
        if isrgb is True:
            pl.imshow(imgdata)
        else:
            MAX_IMG_ROW = 8
            MAX_ROW = 8
            nrow = (channel - 1) / MAX_IMG_ROW + 1
            print '========'
            for i in range( channel):
                pl.subplot(nrow, MAX_IMG_ROW, i + 1)
                curimg = imgproc.imgeq(imgdata[...,i].reshape((s,s)))
                #curimg = imgdata[...,i].reshape((s,s))
                # pl.imshow(curimg, cmap = cm.Greys_r)
                pl.imshow(curimg)
        plt.show()
    def save_AHEBuffy_estimation(self,data_dic, est_pose, matched, save_folder):
        """
        This function require all the buffy image in the
        data_dic['imgdir'] folder
        And save the estimation to save_folder
        """
        import iconvnet_datacvt as icvt
        import iread.buffy as ibuffy
        import Image
        import Stickmen
        from PIL import ImageDraw
        iu.ensure_dir(save_folder)
        
        imgdir = data_dic['imgdir']
        s = np.sqrt(data_dic['data'].shape[0]/3)
        for m in matched:
            i = -m -1 if m < 0 else m
            ep = data_dic['ep'][...,i]
            fr = data_dic['fr'][...,i]
            imgpath = ibuffy.GetImagePath(imgdir, ep,fr)
            img = Image.open(imgpath)
            draw = ImageDraw.Draw(img)
            coor = data_dic['annotation'][...,i]
            Stickmen.draw_sticks(draw, coor, (255,0,0))
            ubd = Stickmen.EstDetFromStickmen(coor)
            draw.rectangle(ubd, outline=(255,0,0))
            if m >= 0:
                bbox = data_dic['oribbox'][...,m]
                
                est_coor = icvt.convert_joints8_to_AHEcoor(est_pose[...,m])
                est_coor = ibuffy.TransformPoints(est_coor.reshape((2,-1),order='F').transpose(), bbox,(s-1,s-1,3),inv=True).transpose().reshape((4,6),order='F')
                Stickmen.draw_sticks(draw, est_coor, (0,255,0))
                det = data_dic['oridet'][...,m]
                draw.rectangle([det[0],det[1],det[2],det[3]], outline=(0,255,0))
                import ipyml.geometry as igeo
                Mkrec = lambda det: ((det[0],det[1]), (det[2], det[3]))
                ri = igeo.RectIntersectRect(Mkrec(det), Mkrec(ubd))
                
            img.save(iu.fullfile(save_folder, str(i) + '.jpg'))
    def transform_pose_back_to_oriimg_coor(self, dp, pose_arr, view_as_train):
        # 
        # dp is data_provider
        import iread
        if 'joint_sample_offset' in self.test_data_provider.data_dic and view_as_train:
            njoints = pose_arr.shape[0]/2
            offset = np.tile(dp.data_dic['joint_sample_offset'], [njoints, 1])
            res_pose = pose_arr  + offset
        else:
            res_pose = pose_arr
            if view_as_train:
                if 'matdim' in self.test_data_provider.batch_meta:
                    to_dim = self.test_data_provider.batch_meta['matdim']
                else:
                    to_dim = (128,128)
            else:
                if 'newdim' in self.test_data_provider.batch_meta:
                    to_dim = self.test_data_provider.batch_meta['newdim']
                else:
                    to_dim = (112,112)
        res_pose = iread.buffy.TransformJoints8Data(res_pose, dp.data_dic['oribbox'], (to_dim[0]-1, to_dim[1]-1) , inv=True)
        return res_pose
        
    def show_joints8_estimation(self):
        import iconvnet_datacvt as icvt
        import iread
        from time import time
        s_time = time()
        data = self.get_next_batch(train=False)[2]
        num_data = data[0].shape[-1]
        
        ests = n.zeros((num_data, 16), dtype=n.single)
        idx = self.joint_idx
        data += [ests]
        img_size = self.test_data_provider.img_size

        self.libmodel.startFeatureWriter(data, idx)
        self.finish_batch()
        print 'It takes %.2f seconds (Including Loading time)' % (time() - s_time) 
        ests = ests.transpose()
        sqdiff = ((data[1] - ests)**2).sum(axis=0)
        all_sqdiff = sqdiff.sum()
        print 'The normalized sqdiff is ' + str( all_sqdiff)
        print 'The normalized avg sqdiff is' + str( all_sqdiff / num_data)
        true_pose = self.test_data_provider.data_dic['joints8'] * img_size
        #true_pose = data[1] * img_size
        est_pose = ests * img_size
        print len(data)
        #print true_pose[:,1].transpose()
        #print est_pose[:,1].transpose()
        estimation_type = op.get_value('estimation_type')
        data_dic = self.test_data_provider.data_dic
        view_as_train = (self.view_as_train == 1)
        if estimation_type is None:
            ######### 
            # Need to convert it into original image
            tpose = self.transform_pose_back_to_oriimg_coor(self.test_data_provider, true_pose, view_as_train)
            epose = self.transform_pose_back_to_oriimg_coor(self.test_data_provider, est_pose, view_as_train)
            #save estimation in original image
            sd = data_dic.copy()
            sd['est_joints8'] = epose
            keys= sd.keys()
            for k in keys:
                if sd[k] is None:
                    del sd[k]
            #icvt.ut.pickle('/home/itsuper7-exp/Desktop/saved-pose', data_dic)
            import scipy.io as sio
            if self.save_estimation:
                sio.savemat(self.save_estimation, sd)
            e = icvt.calc_PCP_from_joints8( tpose, epose, 0.5, True)
        elif estimation_type == 'AHEBuffy':            
            
            Res_list = iread.buffy.MergeAHEBuffyDetection(est_pose, \
                                                    data_dic['oridet'], \
                                                    data_dic['oribbox'],\
                                                    data_dic['annotation'],\
                                                     data_dic['ep'], \
                                                     data_dic['fr'])
            # points will be transformed to original coor
            # in EvaluatePCPFromMergedResult
            dummy1, matched = iread.buffy.EvaluatePCPFromMergedResult(Res_list, \
                                                    (img_size,img_size,3), \
                                                    verbose=True)
            
        # e1 = icvt.calc_Classification_error_from_joints8(true_pose, est_pose, (img_size, img_size, 0))
        save_folder = op.get_value("save_images")
        if save_folder is not None:
            if estimation_type == 'AHEBuffy':
                # data_dic, est_pose, matched, save_folder):
                pickle(iu.fullfile(save_folder, 'matched'), matched)
                self.save_AHEBuffy_estimation(data_dic, est_pose, matched, save_folder)
            else:
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
        
    def save_cost(self):
        # if self.show_cost not in self.train_outputs[0][0]:
        #     raise TestConvNetError('Cost function %s is not defined' % self.show_cost)
        #cost_idxes = [int(x) for x in self.cost_idx.split(',')]
        d = dict()
        d['train_error'] = self.train_outputs
        d['test_error'] = self.test_outputs
        d['costname']= self.train_outputs[0][0].keys()
        d['numbatches'] = len(self.train_batch_range)
        d['testing_freq'] = self.testing_freq
        pickle(self.save_cost_path, d) # 
        
        # for i in cost_idxes:
        #     train_error = [o[0][self.show_cost][i] for o in self.train_outputs]
        #     test_error = [o[0][self.show_cost][i] for o in self.test_outputs]
        #     d['train_error'] += [np.asarray(train_error).reshape((1,-1))]
        #     d['test_error'] += [np.asarray(test_error).reshape((1,-1))]
        # pickle(self.save_cost_path, d)
        
        # print 'There are %d batches' % numbatches
        # print 'len of train_error is %d' % len(train_errors)
        # print 'len of test_error is %d' % len(test_errors)
        # print 'testing freq is %d' % self.testing_freq
        # t = len(train_errors)/( len(test_errors) + 0.0)
        # print 'len train len test = %f' % t
        
    def save_forward_pass_feature(self):
        """
        forward_pass_feature will be in test mode
        
        """
        import iutils as iu
        import scipy.io as sio
        testdp = self.test_data_provider
        num_batches = len(testdp.batch_range)
        print 'There are ' + str(testdp.get_num_batches(self.data_path)) + ' in directory'
        print 'There are ' + str( num_batches ) + ' in range'
        iu.ensure_dir(self.save_feature_path)
        feature_name = self.op.get_value('save_feature_name')
        feature_dim = self.model_state['layers'][self.feature_idx]['outputs']
        for b in range(num_batches):
            epoch, b_num, data = self.get_next_batch(train=False)
            print '   Start writing batch......\t' + str(b_num)
            num_data = data[0].shape[-1]
            data += [n.zeros((num_data, feature_dim), dtype=n.single)]
            save_name = 'data_batch_' + str(b_num)
            save_path = iu.fullfile(self.save_feature_path, save_name)
            self.libmodel.startFeatureWriter(data, self.feature_idx)
            self.finish_batch()
            d = testdp.data_dic.copy()
            d['feature'] = data[-1].transpose()
            d['joints8'] = d['joints8'] * testdp.img_size
            del d['data']
            print 'The shape of feature is' + str(d['feature'].shape)
            pickle(save_path, d)
            sio.savemat(save_path, d)
    def ubd_detect(self):
        # The layer for ubd is specified by self.analyze_feature_name
        # feature_idx is self.feature_idx
        import iutils as iu
        import iread.hmlpe as hp
        ## check self.ubd_dete
        save_folder = self.save_feature_path
        from time import time
        allfile = iu.getfilelist(self.ubd_image_folder, '.*\.(jpg|png|bmp|jpeg)$')
        self.test_data_provider.keep_dic = True
        hmlpe = hp.HMLPE()
        hmlpe.set_imgdata_info({'steps':(28,28)})
        s = 1.618
        scale_pairlist = [ ((1.0/ (s**k)), (1.0/(s**k))) for k in range(0,5)]
        if self.ubd_fix_input_var == -1:
            self.ubd_fix_input_var = 69.8544558 # The mean scale for training  set
        feature_dim = self.model_state['layers'][self.feature_idx]['outputs']
        for f in allfile:
            imgpath = iu.fullfile(self.ubd_image_folder, f)
            t = time()
            d = hmlpe.generate_sliding_detection_data_from_image(imgpath, scale_pairlist)
            ndata = d['data'].shape[-1]
            det_data = np.require(np.zeros((ndata, 1),dtype=np.single), requirements='C')
            det_data = np.require(np.zeros((ndata, feature_dim),dtype=np.single),requirements='C')
            print 'Take %f sec for generating slidding detection' % (time() - t)
            print 'Start Processing: I got %d windows' % ndata
            self.test_data_provider.set_data_dic(d)
            epoch, b_num, data = self.get_next_batch(train=False)
            if self.ubd_fix_input_var:
                cur_s = np.sqrt(np.sum(data[0] ** 2,axis=0)/data[0].shape[0]).reshape((1,ndata))
                print 'median=%.2f mean=%.2f\n======\n' % (np.median(cur_s), np.mean(cur_s))
                cur_s[cur_s == 0] = 1
                data[0] = (data[0]/cur_s) * self.ubd_fix_input_var

            data += [det_data]
            t = time()
            print 'feature idx = %d' % self.feature_idx
            self.libmodel.startFeatureWriter(data,self.feature_idx)
            self.finish_batch()
            # print det_data[12,200:300]
            #print data[0][20:25,10].transpose()
            #t = (det_data.transpose() - data[0]).flatten()
            #print t[:10]
            #print 'diff = %.6f' % (t.sum()/data[0].shape[0]/data[0].shape[1])
            continue
            print 'Need for %f sec for processing' % (time() - t)

            saved = {'detect_score':det_data.transpose(), \
                     'slide_location':d['slide_location'],\
                     'scale':d['scale'],\
                     'imgpath':imgpath}
            pickle(iu.fullfile(save_folder, f + '.detect'), saved)
            print 'Finish %s ' % f
    def get_test_error(self):
        # This will only be used when batch_size is not empty
        # Attention
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        num_cases = []
        while True:
            data = next_data
            num_cases += [data[2][0].shape[-1]]
            self.start_batch(data, train=False)
            next_start_batch_idx = self.test_data_provider.curr_batchnum
            load_next = (not self.test_one) and (next_start_batch_idx != 0)
            if load_next:
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            print "batch %d: %s" % (data[1], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()
        return self.aggregate_test_outputs(test_outputs)
    def norm(self,x):
        return sum(x**2)
    def calc_MPJPE(self, est, gt, num_joints, is_relskel=False):
        """
        est, gt will be dim X ndata matrix
        dim will be dim_data (2 or 3) x num_joints
        """
        ndata = gt.shape[-1]
        est = est.reshape((-1, num_joints, ndata),order='F')
        gt = gt.reshape((-1, num_joints, ndata),order='F')
        print est[:,[0,1,2],0]
        print gt[:,[0,1,2],0]
        return [np.sum(np.sqrt(np.sum((est - gt) ** 2,axis=0)).flatten())/num_joints, ndata]
    def calc_MPJPE_raw(self, est, gt, num_joints, is_relskel=False):
        """
        est, gt will be dim X ndata matrix
        dim will be dim_data (2 or 3) x num_joints
        """
        ndata = gt.shape[-1]
        est = est.reshape((-1, num_joints, ndata),order='F')
        gt = gt.reshape((-1, num_joints, ndata),order='F')
        print est[:,[0,1,2],0]
        print gt[:,[0,1,2],0]
        res = np.sum(np.sqrt(np.sum((est - gt) ** 2,axis=0)),axis=0)/num_joints
        print res.size, ndata
        return res.tolist()
    def calc_absdiff_count(self, est, gt):
        ndata = est.shape[-1]
        print est[0,0:3]
        print gt[0,0:3]
        e = np.sum(np.abs(est - gt).flatten())
        return [e, ndata]
    def convert_pairwise2rel_simple(self, mat):
        import dhmlpe_features
        return dhmlpe_features.convert_pairwise2rel_simple(mat, 3)
    def evaluate_output(self):
        import scipy.io as sio
        next_data=self.get_next_batch(train=False)[2]
        test_outputs = []
        num_cases = []
        params = self.parse_params(self.op.get_value('do_evaluation'))
        output_layer_idx = self.get_layer_idx(params[0])
        if len(params) == 1:
            target_type = 'h36m_body'
            gt_idx = 1
        else:
            target_type = params[1]
            gt_idx = int(params[2])
        data_dim = self.model_state['layers'][output_layer_idx]['outputs']
        test_outputs= []
        tosave_pred = []
        tosave_indexes = []
        err_list = []
        rel_list = ['RelativeSkel_Y3d_mono_body']
        if 'feature_name_3d' not in dir(self.test_data_provider):
            is_relskel = False
        else:
            is_relskel = (self.test_data_provider.feature_name_3d in rel_list)
        print 'I am using %s' % ('RelSkel' if is_relskel else 'Rel')
        convert_dic = {'h36m_rel':lambda x:x,\
                       'h36m_body':self.convert_relskel2rel, \
                       'humaneva_body':self.convert_relskel2rel_eva,
                       'people_count':lambda X: X * self.test_data_provider.maximum_count, \
                       'h36m_pairwise_simple': lambda X: self.convert_pairwise2rel_simple(X)}
        if is_relskel == False and target_type in ['h36m_body']:
            raise Exception('target|dp does''t match')
        while True:
            data = next_data
            num_cases += [data[0].shape[-1]]
            buf = np.require(np.zeros((data[0].shape[-1], data_dim),\
                                      dtype=n.single), \
                             requirements='C')
            data += [buf]
            self.libmodel.startFeatureWriter(data, output_layer_idx)
            cur_batch_indexes = self.test_data_provider.data_dic['cur_batch_indexes']
            next_start_batch_idx = self.test_data_provider.curr_batchnum
            load_next = (not self.test_one) and (next_start_batch_idx!=0)
            if load_next:
                next_data = self.get_next_batch(train=False)[2]
            self.finish_batch()
            if target_type in convert_dic:
                est = convert_dic[target_type](buf.T)
                gt =  convert_dic[target_type](data[gt_idx])
            else:
                est = buf.T
                gt = data[gt_idx]
            if target_type in ['h36m_rel', 'h36m_body', 'humaneva_body', \
                               'h36m_pairwise_simple']:
                test_outputs += [self.calc_MPJPE(est, gt, self.test_data_provider.num_joints)]
                err_list += self.calc_MPJPE_raw(est, gt, self.test_data_provider.num_joints)
            elif target_type == 'h36m_body_len':
                test_outputs += [self.calc_MPJPE(est, gt, self.test_data_provider.num_joints-1)]
            elif target_type == 'people_count':
                test_outputs += [self.calc_absdiff_count(est,gt)]
                err_list += (est - gt).flatten().tolist() 
            print test_outputs[-1]
            if self.save_evaluation:
                tosave_pred += [est]
                tosave_indexes += cur_batch_indexes.flatten().tolist()
            if not load_next:
                break
            sys.stdout.flush()
        a = 0
        b = 0
        for x in test_outputs:
           a = a + x[0]
           b = b + x[1]
        if target_type in ['h36m_rel', 'h36m_body', 'humaneva_body', 'h36m_pairwise_simple']:
            max_depth = self.test_data_provider.max_depth
            print 'max_depth = %6f' % max_depth
            print 'MPJPE is %.6f, a, b = %.6f, %.6f' % ((a/b) * max_depth, a,b)
            arr = np.asarray(err_list).flatten()*max_depth
            print 'MPJPE is %.6f, std =%.6f ' % (np.mean(arr), np.std(arr))
        elif target_type in [ 'people_count']:
            print 'Average counting error is %.6f (%d patches)' % (a/b, b)
            err_arr = np.abs(np.asarray(err_list))
            print 'Average counting error is %.6f (std=%.6f)' % (np.mean(err_arr), \
                                                                 np.std(err_arr))
        if self.save_evaluation:
            saved = dict()
            if target_type in ['h36m_body', 'humaneva_body','h36m_rel', \
                               'h36m_pairwise_simple']:
                saved['prediction'] = np.concatenate(tosave_pred, axis=-1) * self.test_data_provider.max_depth
            else:
                saved['prediction'] = np.concatenate(tosave_pred, axis=-1)
            saved['indexes'] = tosave_indexes
            if len(self.test_data_provider.images_path) != 0:
                saved['images_path'] = [self.test_data_provider.images_path[x] for x in tosave_indexes]
            saved['oribbox'] = self.test_data_provider.batch_meta['oribbox'][...,tosave_indexes].reshape((4,-1),order='F')
            sio.savemat(self.save_evaluation, saved)
    def do_score_prediction(self):
        """
        IN the current version, I will not take parameters from outside
        THis will be improved in the future.
        """
        
        import iread.myio as mio
        from igui.score_canvas import ScoreCanvas
        exp_name = 'JPS_act_12_exp_4_accv_half_fc_j2'
        exp_name = 'JPS_act_14_exp_2_accv'
        exp_name_base = 'ASM_act_14_exp_2'
        exp_base_folder = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp'
        exp_path = iu.fullfile(exp_base_folder, 'folder_%s' % exp_name, 'batches.meta')
        meta_base_path = iu.fullfile(exp_base_folder, 'folder_%s' % exp_name_base, 'batches.meta')
        meta = mio.unpickle(exp_path)
        meta_base = mio.unpickle(meta_base_path)
        images_path = meta_base['images_path']
        
        pred_pose = meta['feature_list'][0]
        gt_pose = meta['random_feature_list'][0]
        ntotal = gt_pose.shape[-1]
        print 'gt_pose_shape',gt_pose.shape
        print 'pred_pose_shape', pred_pose.shape
        ref_frame = 3600 # This is the index in test range
        ## ref_frame = 2600 # This is the index in test range
        test_range = self.test_data_provider.feature_range
        ref_idx = test_range[ref_frame]
 
        n_to_show = 1000
        
        idx_to_show = np.random.choice(ntotal, n_to_show - 1)
        idx_to_show = [ref_idx] + idx_to_show.tolist()  
        idx_to_show = np.asarray(idx_to_show, dtype=np.int).flatten()
        
        ref_pose =  pred_pose[...,ref_idx].reshape((-1,1),order='F')      
        pose_to_eval =gt_pose[...,idx_to_show]
        output_feature_name = 'fc_2' # <------------------Parameter
        output_layer_idx = self.get_layer_idx(output_feature_name)

        # do it once <------------- Maybe it can support multiple batch in the future
        data_dim = self.model_state['layers'][output_layer_idx]['outputs']
        print 'data_dim', data_dim
        
        cur_data = [np.require(np.tile(ref_pose, [1,n_to_show]), \
                               dtype=np.single,requirements='C'), \
                    np.require(pose_to_eval.reshape((-1,n_to_show),order='F'),\
                               dtype=np.single,requirements='C'), \
                    np.require(np.zeros((1,n_to_show),dtype=np.single), \
                               requirements='C'),
                    np.require(np.zeros((n_to_show,data_dim),dtype=np.single), \
                               requirements='C')]
        residuals = cur_data[1][...,0].reshape((-1,1),order='F') - cur_data[1]
        dp = self.test_data_provider
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, dp.num_joints)

        gt_score = dp.calc_score(mpjpe, dp.mpjpe_factor/dp.max_depth,\
                              dp.mpjpe_offset/dp.max_depth).reshape((1,n_to_show)).flatten()
        self.libmodel.startFeatureWriter(cur_data, output_layer_idx)
        self.finish_batch()
        score = cur_data[-1].T
        print 'dim score', score.shape, 'dim gt_score', gt_score.shape
        score = score.flatten()
        # score = gt_score.flatten()
        def my_sort_f(k):
            if k == 0:
                return 10000000
            else:
                return score[k]
        sorted_idx = sorted(range(n_to_show), key=my_sort_f,reverse=True)
        s_to_show = [idx_to_show[k] for k in sorted_idx]
        sorted_score = np.asarray( [score[k] for k in sorted_idx])
        
        pose_to_plot = self.convert_relskel2rel(cur_data[1])
        sorted_pose = pose_to_plot[...,sorted_idx]
        class ScorePoseCanvas(ScoreCanvas):
            def __init__(self,data_dic):
                import iread.h36m_hmlpe as h36m
                ScoreCanvas.__init__(self,data_dic)
                self.pose_data = data_dic['pose_data']
                self.limbs = h36m.part_idx
                self.tmp = 0
            def show_image(self,ax):
                # ScoreCanvas.show_image(self,ax)
                # return
                import Image
                idx =self.cur_data_idx
                if idx == 0:
                    self.tmp = self.tmp + 1
                    if self.tmp == 1:
                        img = self.load_image(idx)
                        ax.imshow(np.asarray(img))
                        return
                print 'Current data idx %d ' % self.cur_data_idx
                # params = {'elev':-89, 'azim':-107}
                # params = {'elev':-69, 'azim':-107}
                params = {'elev':-81, 'azim':-91} # frontal view
                fig = plt.figure(100)
                from mpl_toolkits.mplot3d import Axes3D
                import imgproc
                # new_ax = self.fig.add_axes( rng_rel,projection='polar')
                new_ax = fig.add_subplot(111,projection='3d')
                imgproc.turn_off_axis(new_ax)
                cur_pose = self.pose_data[...,idx].reshape((3,-1),order='F')
                dutils.show_3d_skeleton(cur_pose.T,\
                                        self.limbs, params)
                xmin,xmax = np.min(cur_pose[0]),np.max(cur_pose[0])
                ymin,ymax = np.min(cur_pose[1]),np.max(cur_pose[1])
                zmin,zmax = np.min(cur_pose[2]),np.max(cur_pose[2])
                def extent(x,y,ratio):
                    x = x + (x-y) * ratio
                    y = y + (y-x) * ratio
                    return -0.4,0.4
                r = 0.1
                new_ax.set_xlim(extent(xmin,xmax,r))
                new_ax.set_ylim(extent(ymin,ymax,r))
                new_ax.set_ylim(extent(zmin,zmax,r))
                tmp_folder = '/public/sijinli2/ibuffer/2014-CVPR2015/tmp/images'
                save_path = iu.fullfile(tmp_folder, 'tmp_image.png')
                plt.savefig(save_path)
                img = Image.open(save_path)
                plt.close(100)
                img_arr = np.asarray(img)
                s = np.int(img_arr.shape[0]/5.0)
                e = np.int(img_arr.shape[0] - s)
                s  = 0
                e = img_arr.shape[0]
                img_arr = img_arr[s:e,:,:]
                ax.imshow(np.asarray(img_arr))
                # ax.plot([1,0,0],[0,1,0],[0,0,1])


        sc = ScorePoseCanvas({'x': np.asarray(range(len(idx_to_show))), 'y':sorted_score,\
                          'images_path': [images_path[k] for k in s_to_show], \
                          'pose_data':sorted_pose})
        sc.start()
        print 'max score is ' , sorted_score.max()
        gt_sort_idx = sorted(range(n_to_show), key=lambda k:gt_score[k], reverse=True)
        sorted_gt_score = np.asarray([gt_score[k] for k in gt_sort_idx])
        sorted_score_by_gt = [score[k] for k in gt_sort_idx]
        pl.plot(np.asarray(range(n_to_show)), sorted_gt_score, 'r', label='gt_score')
        pl.plot(np.asarray(range(n_to_show)), sorted_score_by_gt, 'g', label='pred_score')
        pl.legend()
    
    def start_SLP_server(self):
        """
        This function is used for structure learning project.
        It will read data from the inbox and produce file in outbox
        """
        import slp
        
        s = self.op.get_value('mode_params')
        if len(s) == 0:
            print 'No parameter received'
            params = {}
        else:
            l_p = self.parse_params(s)
            params = {'action':[np.int(l_p[0])]}
            if len(l_p) == 2:
                params['image_feature_layer_name'] = l_p[1]
        slpserver = slp.SLPServerTrial(self, params)
        slpserver.start()
    def get_backtrack_layer_list(self, layer_idx):
        """
        This function can be used for generating the back trackted layer list
        """
        res_list =  [self.model_state['layers'][layer_idx]]
        while ('inputLayers' in res_list[0]):
            ### If there are multiple input, take the 0-th layer
            l = res_list[0]['inputLayers'][0]
            if l['type'] == 'data':
                break
            res_list = [l] + res_list
        return res_list
    def start(self):
        self.op.print_values()
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        if self.mode:
            if self.mode == 'do-score-prediction':
                self.do_score_prediction()
            elif self.mode == 'slp-server':
                self.start_SLP_server()
        if self.analyze_output:
            self.show_prediction()
        if self.show_estimation:
            self.show_joints8_estimation()
        elif self.analyze_feature_name:
            print type(self.ubd_image_folder)
            if self.ubd_image_folder:
                self.ubd_detect()
            else:
                self.analyze_feature() 
        elif self.save_feature_name:
            if self.forward_pass_feature:
                self.save_forward_pass_feature()
            else:
                self.save_feature()
        elif self.save_cost_path:
            self.save_cost()
        elif self.do_evaluation:
            self.evaluate_output()
        elif self.show_cost:
            self.plot_cost()    
        plt.show()        
        sys.exit(0)
    @classmethod
    def convert_relskel2rel(cls, x):
        import dhmlpe_features as df
        return df.convert_relskel2rel(x)
    def convert_relskel2rel_eva(cls,x):
        import dhmlpe_features as df
        import humaneva_meta as hm
        return df.convert_relskel2rel_base(x, hm.limbconnection)
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'data_path', 'minibatch_size', 'layer_params', 'batch_size', 'test_only', 'test_one', 'shuffle_data', 'crop_one_border', 'external_meta_path'):
                op.delete_option(option)
        op.add_option("analyze-output", "analyze_output", StringOptionParser, "Show specified objective function")
        op.add_option("label-idx", "label_idx", IntegerOptionParser, "The layer idx, with which the output compare") 
        op.add_option("show-estimation", "show_estimation", StringOptionParser, "Show pose estimation result")
        op.add_option("estimation-type", "estimation_type", StringOptionParser, "Determine What type of estimation to use, joints8(default or AHE)")
        op.add_option("save-estimation", "save_estimation", StringOptionParser, "save the estimation result in .mat")
        op.add_option("show-images", "show_images", StringOptionParser, "Whether to use the estimated images")
        op.add_option("save-images", "save_images", StringOptionParser, "Save the estimated images")
        op.add_option("calc-hist", "calc_hist", StringOptionParser, "Calculate histogram based stochastics")
        op.add_option("save-feature-name", 'save_feature_name', StringOptionParser, "Save features in layers specified in save_features")
        op.add_option('forward-pass-feature', 'forward_pass_feature', IntegerOptionParser, "indicate whether to transform feature")
        op.add_option('save-feature-path', 'save_feature_path', StringOptionParser, "save layer feature in 'save_feature_path' ")
        op.add_option('analyze-feature-name', 'analyze_feature_name', StringOptionParser, "The layer name of the feature to be analyzed")
        op.add_option('show-response', 'show_response', StringOptionParser, "Displaying the response of input images, used with analyze-feature-name")
        op.add_option('save-response', 'save_response', StringOptionParser, "Save image response of input images, used with analyze-feature-name")
        op.add_option('save-res-patch', 'save_res_patch', StringOptionParser, "Fine the patches responsbile for high activation")
        op.add_option('save-indmap-show', 'save_indmap_show', StringOptionParser, "Display indicator map alone with original image")
        op.add_option('matched-path', 'matched_path', StringOptionParser, "The file for removing all unmatched result")
        op.add_option('feature-channel', 'feature_channel', IntegerOptionParser, "The channel of features")
        op.add_option('ubd-image-folder', 'ubd_image_folder', StringOptionParser, "The folder with images to be analyzed for upper body detection")
        op.add_option('ubd-fix-input-var', 'ubd_fix_input_var', FloatOptionParser, "scale data so that the np.sum((x - mean)**2) == constant")
        # op.add_option('show-cost', 'show_cost', StringOptionParser, 'Display Costs during training')
        # op.add_option("cost-idx", "cost_idx", StringOptionParser, "Cost function return value index for --show-cost", default='0')
        op.add_option('ind-type', 'ind_type', StringOptionParser, 'Indicating whether this is part or joint indicator')
        op.add_option('save-cost-path', 'save_cost_path', StringOptionParser, 'the path to save costs')
        op.add_option("view-as-train", 'view_as_train', IntegerOptionParser, "When regard evaluted data as training batch")
        op.add_option('do-evaluation', 'do_evaluation', StringOptionParser, 'Do seperate evaluation steps')
        op.add_option('evaluation-type', 'evaluation_type', StringOptionParser, 'Specify which kinds of measure will be used')
        op.add_option('save-evaluation', 'save_evaluation', StringOptionParser, 'The folder for saving evaluation results')
        op.add_option('show-cost', 'show_cost', StringOptionParser, 'The name of cost to show')
        op.add_option('cost-idx', 'cost_idx', IntegerOptionParser, 'The index of cost for showing --show-cost', default=0)
        op.add_option('mode', 'mode', StringOptionParser, 'Determine what to do next')
        op.add_option('mode-params', 'mode_params', StringOptionParser, 'Determine what to do next', default='')
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
