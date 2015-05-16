from init_test import *
import dhmlpe_utils as dutils
import iutils as iu
import scipy.io as sio
import iread.myio as mio 
cutils = dutils.Cutils()
dskels = sio.loadmat('/opt/visal/data/H36/H36MData/SLP/data/AngleSamples/skel2.mat')
skel = dskels['skel']
print dskels.keys()
dpose = mio.unpickle('/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_2/batches.meta')
gt_pose = dpose['Relative_Y3d_mono_body']
print gt_pose.shape
dangle = sio.loadmat('/opt/visal/data/H36/H36MData/SLP/data/AngleSamples/ASM_act_14_angles.mat')
print dangle.keys()
print dangle['angles_range'].shape
t = dangle['angles_range'].flatten()
print max(t), min(t)
print dangle['angles'].shape
gt_angle = dangle['angles'][0,:,:]
[g_pose,g_rot] = cutils.convert_angle2(skel, gt_angle)

