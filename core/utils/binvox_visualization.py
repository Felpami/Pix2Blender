# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
import time

import matplotlib
matplotlib.use('wxAgg')
import matplotlib.pyplot as plt

import os
from mpl_toolkits.mplot3d import Axes3D

def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Pix2Vox Preview')
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('auto')
    ax.voxels(volume, edgecolor="k")
    plt.show()
    plt.close()
    return
