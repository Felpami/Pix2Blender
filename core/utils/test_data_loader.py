import queue

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset
import time
import matplotlib.pyplot as plt

from datetime import datetime as dt
from enum import Enum, unique

import utils.binvox_rw

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, n_views_rendering, transforms=None):
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rendering_images = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return rendering_images

    def get_datum(self, idx):
        self.n_views_rendering = len(self.file_list[idx]['rendering_images'])
        rendering_image_paths = self.file_list[idx]['rendering_images']


        # Get data of rendering images
        selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = plt.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                raise Exception('[FATAL] %s It seems that there is something wrong with the image file %s' % (dt.now(), image_path))

            rendering_images.append(rendering_image)
        # Get data of volume
        return np.asarray(rendering_images)


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class DataLoader():
    def __init__(self):
        print('[INFO] %s Starting Process.' % (dt.now()))

    def get_dataset(self, path, extension, n_views_rendering, transforms=None):
        files = []
        files.extend(self.get_images(path, extension))

        print('[INFO] %s Complete collecting images from %s. Total images files = %d.' % (dt.now(), path, len(files[0]['rendering_images'])))
        return Dataset(files, n_views_rendering, transforms)

    def get_images(self, path, extension):
        images= []

        # Get file list of rendering images
        img_file_path = path + '\\%02d.%s' % (0, extension)
        try:
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
        except:
            raise Exception('[Fatal] %s %s directory does not exist.' % (dt.now(), path))

        rendering_image_indexes = range(total_views)
        rendering_images_file_path = []
        for image_idx in rendering_image_indexes:
            img_file_path = path + '\\%02d.%s' % (image_idx, extension)
            if not os.path.exists(img_file_path):
                continue
            print('[INFO] %s Found image = %s.' % (dt.now(), img_file_path))
            rendering_images_file_path.append(img_file_path)

        if len(rendering_images_file_path) == 0:
            raise Exception('[Fatal] %s No images files found in = %s.' % (dt.now(), path))

        # Append to the list of rendering images
        images.append({
            'rendering_images': rendering_images_file_path,
        })

        return images