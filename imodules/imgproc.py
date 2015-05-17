
"""
Written by Li Sijin

Some common image processing functions
"""
import numpy as np
import matplotlib.pyplot as plt

def graytoheapimage(img):
    cmap = plt.get_cmap('jet')
    return np.delete(cmap(img), 3, 2)
def ensure_rgb(img):
    if len(img.shape) == 2 or len(img.shape)==3 and img.shape[-1]==1:
        img = img.reshape((img.shape[0],img.shape[1],-1))
        img = np.concatenate((img,img,img),axis=-1)
    elif len(img.shape)==3 and img.shape[-1]==3:
        pass
    elif len(img.shape)==3:
        img = img[...,0:3]
    return img
def get_color_list(N):
    """
    This function will reburn a list of n different colors
    each element is represented by (r,g,b) tuple
    """
    import colorsys
    HSV_tuples = [(x*1.0/N, 0.9, 0.9) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples
def set_figure_size(width, height):
    from pylab import rcParams
    rcParams['figure.figsize'] = width, height