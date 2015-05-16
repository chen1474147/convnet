"""
Usage:
examples
python demo_hmlpe_pose.py -f /opt/visal/tmp/for_sijin/Data/saved/backup/ACCV2014/c8k16/ConvNet__2014-05-19_15.32.55 --do-pose-evaluation=fc_j2 --inputstream=imgcamera,/public/sijinli2/ibuffer/2014-08-08/upperbody-bkg2 --outputdir=/public/sijinli2/ibuffer/2014-08-08/upperbody-bkg2_output


python demo_hmlpe_pose.py -f /opt/visal/tmp/for_sijin/Data/saved/backup/IJCV2014/c8k17/ConvNet__2014-01-31_17.22.34_backup --do-pose-evaluation=fc_j2,hmlpe_2d,1 --inputstream=imgcamera,/public/sijinli2/ibuffer/2014-08-08/public/sijinli2/ibuffer/2014-08-08/upperbody2d



folders for test

/opt/visal/data/H36/H36MData/Train_square/s_01_act_02_subact_01_ca_02



/opt/visal/tmp/for_sijin/Data/saved/backup/ACCV2014/c8k19/ConvNet__2014-06-16_18.30.15

walking dog
/opt/visal/tmp/for_sijin/Data/saved/backup/ACCV2014/c8k14/ConvNet__2014-06-17_21.33.01

/opt/visal/tmp/for_sijin/Data/saved/backup/ACCV2014/c8k16/ConvNet__2014-05-19_15.32.55
Eating
/opt/visal/tmp/for_sijin/Data/saved/c8k18/ConvNet__2014-06-13_20.27.48/opt/visal/tmp/for_sijin/Data/saved/c8k18/ConvNet__2014-06-13_20.27.48
*test for 2d cases*
+ /opt/visal/tmp/for_sijin/Data/saved/backup/IJCV2014/c8k17/ConvNet__2014-01-31_17.22.34_backup
+ 

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
import iutils as iu
import Image
import matplotlib.animation as animation
import iread.h36m_hmlpe as h36m
import ImageDraw
from time import time
import iread.hmlpe as hmlpe
sys.path.append('/home/grads/sijinli2/Projects/DHMLPE/Python/src/')
sys.path.append('/media/M_FILE/cscluster/Projects/DHMLPE/Python/src/')
class DemoError(Exception):
    pass
class ICameraBasic:
    def __init__(self):
        self.is_open = True
    def get_frame(self):
        if self.is_open == False:
            raise DemoError('Camera is not openned')
    def destroy(self):
        self.is_open =  False
class ImageCamera(ICameraBasic):
    def __init__(self, imgdir):
        self.Image = __import__('Image')
        self.imgdir = imgdir
        self.cur_idx = -1
        self.images_path = [iu.fullfile(imgdir, x) for x in \
                            sorted(iu.getfilelist(imgdir, '.*\.(jpg|png)'))]
        if len(self.images_path) == 0:
            raise DemoError('I cannot find image uder %s ' % self.images_path)
        print 'I got %d images' % len(self.images_path)   
        ICameraBasic.__init__(self)
    def get_frame(self):
        self.cur_idx = self.cur_idx + 1
        if self.cur_idx >= len(self.images_path):
            self.cur_idx = self.cur_idx % len(self.images_path)
        return True, np.asarray(self.Image.open(self.images_path[self.cur_idx]))
class CVCamera(ICameraBasic):
    def __init__(self):
        self.cv2 = __import__('cv2')
        self.cap = self.cv2.VideoCapture(0)
        ICameraBasic.__init__(self)
    def get_frame(self):
        ICameraBasic.get_frame(self)
        ret, frame = self.cap.read()
        return ret, frame[:,::-1,[2,1,0]]
    def destroy(self):
        ICameraBasic.destroy(self)
        self.cap.release()
class ICropImage:
    def __init__(self):
        pass
    def get_bbox(img):
        pass
    def destroy(self):
        pass
class IFaceUbdCropImage(ICropImage):
    """
    Bounding box in x,y, w,h format
    """
    def __init__(self, options = None):
        self.cv2 = __import__('cv2')
        self.igeo = __import__('ipyml').geometry
        if options is None:
            self.model_dir =  '/opt/visal/tmp/for_sijin/Data/opencv_trained_models'
        else:
            self.model_dir = options['model_dir']
        self.face_cascade = dict()
        self.face_cascade['frontal'] = self.cv2.CascadeClassifier(iu.fullfile(self.model_dir, \
                                                                   'haarcascade_frontalface_default.xml'))
        self.face_cascade['profile'] = self.cv2.CascadeClassifier(iu.fullfile(self.model_dir, \
                                                                           'haarcascade_profileface.xml'))
    def face2bnd(self, bnd_list, imgsize):
        r = []
        for p in bnd_list:
            x,y,w,h = p[0],p[1],p[2],p[3]
            nw = w / 20 * 180
            nh = h / 20 * 120
            nx = max(0,x - w / 10 * 130)
            ny = max(0,y - h / 2)
            nw = min(imgsize[1] - 1 - nx, nw)
            nh = min(imgsize[0] - 1 - ny, nh)
            r = r + [(nx,ny,nw,nh)]
        return r
    def get_bbox(self,img):
        gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        allfaces = []
        t1 = time()
        for x in self.face_cascade:
            faces = self.face_cascade[x].detectMultiScale(gray, 1.3, 5) 
            allfaces += [ tuple(x.flatten()) for x in faces ]
        return self.face2bnd(self.do_nms(allfaces), img.shape)
    def do_nms(self,face_list):
        re_list = []
        for x,y,w,h in face_list:
            add_flag = True
            if len(re_list) != 0:
                for xx,yy,ww,hh in face_list:
                    int_rect = self.igeo.RectIntersectRect( ((x,y),(x+w,x+h)), ((xx,yy),(xx+ww,yy+hh)) )
                    if len(self.int_rect) == 0 or \
                      (int_rect[2] - int_rect[0]+1)*(int_rect[3] - int_rect[1]+1) < (ww + 1)*(hh+1)*0.5:
                        continue
                    add_flag = False
                    break
            if add_flag:
                re_list += [(x,y,w,h)]
        return re_list                    
class TestConvNet(ConvNet):
    def __init__(self, op, load_dict):
        ConvNet.__init__(self, op, load_dic)
        self.statistics = dict()
        self.temp_data = dict()
        self.cropper_dict = {'faceubd':IFaceUbdCropImage}
    def get_gpus(self):
        self.need_gpu = self.op.get_value('do_pose_evaluation') is not None
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
        pass
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
    def plot_skelenton(self, v, connections, ax):
        v = v.reshape((3,-1),order='F')
        for p in connections:
            ax.plot( v[0,[p[0],p[1]]], v[1, [p[0],p[1]]], v[2, [p[0], p[1]]])
    def draw2d_skelenton(self, vec, connections, draw):
        ndata = vec.shape[-1]
        for i in range(ndata):
            v = vec[...,i].reshape((2,-1),order='F')
            for p in connections:
                draw.line((v[0,p[0]], v[1,p[0]], v[0,p[1]], v[1,p[1]]),\
                          fill=(255,0,0), width=3)
    def estimate_pose_main_process(self, input_dic, output_dic):
        import imgproc
        load_next = True
        data, raw_img = input_dic['next_data'], input_dic['raw_img']
        ndata = data[0].shape[-1]
        # output_dic['num_cases'] += [data[0].shape[-1]]
        buf = np.require(np.zeros((data[0].shape[-1], input_dic['data_dim']),\
                                      dtype=n.single), \
                                requirements='C')
        data += [buf]
        start_t = time()
        self.libmodel.startFeatureWriter(data, input_dic['output_layer_idx'])
        if load_next:
            input_dic['next_data'], input_dic['raw_img'], input_dic['bnd'] = \
              self.get_hmlpe_posedata_from_camera(input_dic['camera'], self.test_data_provider)
        self.finish_batch()
        print '[%.6f seconds]' % (time() - start_t)
        if input_dic['target_type'] in input_dic['convert_dic']:
            output_dic['est'] = input_dic['convert_dic'][input_dic['target_type']](buf.T)
        else:
            output_dic['est'] = buf.T
        if not load_next:
            return
        sys.stdout.flush()
        tmp = input_dic['raw_img']
        input_dic['camera_im'].set_data(tmp)
        input_dic['camera_fig'].canvas.draw()
        s = int(np.sqrt(data[0].shape[0]))
        if input_dic['target_type'] == 'hmlpe_2d':
            img = Image.fromarray(np.require(input_dic['raw_img'],dtype=np.uint8))
            sx,sy = data[0].shape[1], data[0].shape[0]
            output_dic['est'] = output_dic['est'].reshape((2,-1,ndata), order='F')
            njoints = output_dic['est'].shape[1]
            cur_bnd = input_dic['bnd']            
            bnd_sx = np.tile(np.asarray([(v[2]+1.0) \
                                          for v in cur_bnd]).reshape((1,ndata)),\
                              (njoints, 1)).reshape((1,njoints,ndata),order='F')
            bnd_sy = np.tile(np.asarray([(v[3]+1.0) \
                                          for v in cur_bnd]).reshape((1,ndata)),\
                              (njoints, 1)).reshape((1,njoints,ndata),order='F')
            bnd_s = np.concatenate((bnd_sx,bnd_sy),axis=0)

            bnd_ax = np.tile(np.asarray([v[0]  \
                                         for v in cur_bnd]).reshape((1,ndata)),\
                              (njoints, 1)).reshape((1,njoints,ndata),order='F')
            bnd_ay = np.tile(np.asarray([v[1] \
                                          for v in cur_bnd]).reshape((1,ndata)),\
                              (njoints, 1)).reshape((1,njoints,ndata),order='F')
            bnd_a = np.concatenate((bnd_ax,bnd_ay),axis=0)
            output_dic['est'] = output_dic['est'] * bnd_s + bnd_a

            draw = ImageDraw.Draw(img)
            # draw bnd
            for b in cur_bnd:
                draw.rectangle((b[0],b[1],b[0]+b[2],b[1]+b[3]))
            # draw center rectangle
            hx,hy = img.size 
            draw.rectangle((hx/2 - hy/2,0, hx/2+hy/2, hy ), outline=(255,0,0))
            self.draw2d_skelenton(output_dic['est'], hmlpe.part_idx, draw)
            input_dic['pose_ax'].set_data(np.asarray(img))
            input_dic['pose_fig'].canvas.draw()
        else:
            ## Plot pose
            input_dic['pose_ax'].cla()
            #input_dic['pose_ax'].view_init(azim=-92, elev=-46)
            vlim = 0.4
            input_dic['pose_ax'].set_xlim([-vlim,vlim])
            input_dic['pose_ax'].set_ylim([-vlim,vlim])
            input_dic['pose_ax'].set_zlim([-vlim,vlim])
            self.plot_skelenton(output_dic['est'], h36m.part_idx, input_dic['pose_ax'])
            imgproc.turn_off_axis(input_dic['pose_ax'])
            input_dic['pose_fig'].canvas.draw()
        if 'outputdir' in input_dic:
            outputdir = input_dic['outputdir']
            savecnt = input_dic['savecnt']
            print outputdir
            for i in range(2):
                plt.figure(i)
                plt.savefig(iu.fullfile(outputdir, 'fig_%02d_%06d.jpg' % (i,savecnt)))
            input_dic['savecnt'] = savecnt + 1
        return input_dic['camera_im'], input_dic['pose_ax']
    def estimate_pose(self):
        import scipy.io as sio
        from mpl_toolkits.mplot3d import Axes3D
        self.crop_cls = None
        if self.op.get_value('crop_image'):
            if self.op.get_value('crop_image') in self.cropper_dict:
                self.crop_cls = self.cropper_dict[self.op.get_value('crop_image')]()
        params = self.parse_params(self.op.get_value('do_pose_evaluation'))
        input_params = self.parse_params(self.op.get_value('inputstream'))
        input_type = str(input_params[0])
        if input_type == 'imgcamera':
            ca = ImageCamera(str(input_params[1]))
        else:
            ca = CVCamera()
        output_layer_idx = self.get_layer_idx(params[0])
        if len(params) == 1:
            target_type = 'h36m_body'
            gt_idx = 1
        else:
            target_type = params[1]
            gt_idx = int(params[2])
        data_dim = self.model_state['layers'][output_layer_idx]['outputs']
        if 'feature_name_3d' not in dir(self.test_data_provider):
            is_relskel = False
        else:
            is_relskel = (self.test_data_provider.feature_name_3d == 'RelativeSkel_Y3d_mono_body')
        print 'I am using %s' % ('RelSkel' if is_relskel else 'Rel')
        convert_dic = {'h36m_body':self.convert_relskel2rel, \
                       'humaneva_body':self.convert_relskel2rel_eva}
        input_dic = {'data_dim':data_dim, 'target_type':target_type, \
                     'output_layer_idx':output_layer_idx}
        output_dic = {'est':None}
        input_dic['convert_dic'] = convert_dic
        input_dic['camera'] = ca
        input_dic['next_data'], input_dic['raw_img'], input_dic['bnd'] = \
          self.get_hmlpe_posedata_from_camera(input_dic['camera'], self.test_data_provider)
        input_dic['camera_fig'] = plt.figure(0)
        input_dic['camera_im'] = plt.imshow(input_dic['raw_img'])
        input_dic['pose_fig'] = plt.figure(1)
        if target_type == 'hmlpe_2d':
            input_dic['pose_ax'] = plt.imshow(input_dic['raw_img'])
        else:
            input_dic['pose_ax'] = input_dic['pose_fig'].add_subplot(111,projection='3d')
            input_dic['pose_ax'].plot(range(10),range(10),range(10))
            input_dic['pose_ax'].view_init(azim=-94, elev=-71)
        if self.op.get_value('outputdir'):
            input_dic['outputdir'] = self.op.get_value('outputdir')
            input_dic['savecnt'] = 0
            iu.ensure_dir(input_dic['outputdir'])
        ani_func = lambda *x: self.estimate_pose_main_process(input_dic, output_dic)
        dummy = animation.FuncAnimation(input_dic['camera_fig'], ani_func, \
                                interval=5, blit=True, repeat=False)
        plt.show()        
    def get_hmlpe_posedata_from_camera(self,ca, test_dp):
        dummy, raw_frame = ca.get_frame()
        if self.crop_cls is None:
            bnd = []
        else:
            t1 = time()
            bnd = self.crop_cls.get_bbox(raw_frame)
            print 'bnd', bnd, 'cost %.6f' % (time() - t1)
        if bnd is None or len(bnd) == 0:
            bnd =  [(0,0, raw_frame.shape[1]-1, raw_frame.shape[0]-1)]
        ndata = len(bnd)
        if 'cropped_mean_image' in dir(test_dp):
            sp = test_dp.cropped_mean_image.shape
            mean_image = test_dp.cropped_mean_image
        else:
            sp = [test_dp.img_size,test_dp.img_size]
            mean_image = test_dp.data_mean
        frame = np.zeros((sp[0] * sp[1] * 3,ndata))
        i = 0
        for (x,y,w,h) in bnd:
            img = Image.fromarray(raw_frame[x:x+w+1,y:y+w+1,:]).resize((sp[1],sp[0]))
            frame[:,i] = (np.asarray(img).reshape((-1,1),order='F') \
                                - mean_image.reshape((-1,1),order='F')).flatten()
            i = i + 1
        frame = np.require(frame, dtype=np.single, requirements='C')
        pose = np.require(np.ndarray((test_dp.get_data_dims(1),ndata), order='F'), dtype=np.single, requirements='C')
        ind = np.require(np.ndarray((test_dp.get_data_dims(2),ndata), order='F'), dtype=np.single, requirements='C')
        if 'jt_inddim' in dir(test_dp):
            ind_jt = np.require(np.ndarray((test_dp.get_data_dims(3),ndata), order='F'), dtype=np.single, requirements='C')
            mask =  np.require(np.ndarray((test_dp.get_data_dims(4),ndata), order='F'), dtype=np.single, requirements='C')
            is_pos = np.require(np.ndarray((test_dp.get_data_dims(5),ndata), order='F'), dtype=np.single, requirements='C')
            return [frame, pose, ind, ind_jt, mask, is_pos], raw_frame, bnd
        else:
            return [frame, pose, ind], raw_frame        
    def start(self):
        self.op.print_values()
        if self.do_pose_evaluation:
            self.estimate_pose()
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
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'data_path', 'minibatch_size', 'layer_params', 'batch_size', 'test_only', 'test_one', 'shuffle_data', 'crop_one_border'):
                op.delete_option(option)
        op.add_option('do-pose-evaluation', 'do_pose_evaluation', StringOptionParser, 'Specify the output layer of pose')
        op.add_option('inputstream', 'inputstream', StringOptionParser, 'Specify the type of camera to use [imgcamera|cvcamera]')
        op.add_option('outputdir', 'outputdir', StringOptionParser, 'Specify the directory for saving outputs')
        op.add_option('crop-image', 'crop_image', StringOptionParser, 'Specify the method to crop image to square input patch [faceubd]') 
        op.options['load_file'].default = None
        return op

if __name__ == "__main__":
    try:
        op = TestConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = TestConvNet(op, load_dic)
        model.start()
    except (UnpickleError, DemoError, opt.GetoptError), e:
        print '-----------------'
        print "error"
        print e
        print '           '
