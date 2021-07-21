# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import platform
import os
import sys
import subprocess

import pkg_resources
from datetime import datetime as dt
import time

with open('.\\requirements.txt') as f:
    package = f.read().splitlines()

for pack in package:
    try:
        pkg_resources.get_distribution(pack)
    except pkg_resources.DistributionNotFound:
        try:
            print('[INFO] %s Installing package : %s.' % (dt.now(), pack))
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pack, '--user'])
            print('[INFO] %s Successfully installed : %s.' % (dt.now(), pack))
        except:
            raise Exception('[FATAL] %s Something went wrong during the installation of : %s.' % (dt.now(), pack))

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision

from torchvision.transforms import ToTensor
from numpy import asarray
from argparse import ArgumentParser
from pyvox.models import Vox
from pyvox.writer import VoxWriter

from config import cfg
import utils.test_data_loader
import utils.binvox_visualization
import utils.data_transforms
import utils.network_utils

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

# Unlock the main function

def test_net(cfg,
             path=None,
             test_data_loader=None,
             plot=False,
             extension=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None
             ):

    print('[INFO] %s Setting up DataLoader.' % (dt.now()))

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        #CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.ImageResize(IMG_SIZE),
            #utils.data_transforms.CenterCrop(IMG_SIZE,CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.test_data_loader.DataLoader()
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(path, extension, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                batch_size=1,
                                                num_workers=1,
                                                pin_memory=True,
                                                shuffle=False)

        print('[INFO] %s Setting up Network.' % (dt.now()))

        # Set up networks
        if (torch.cuda.is_available()):
            encoder = torch.nn.DataParallel(Encoder(cfg)).cuda()
            decoder = torch.nn.DataParallel(Decoder(cfg)).cuda()
            refiner = torch.nn.DataParallel(Refiner(cfg)).cuda()
            merger = torch.nn.DataParallel(Merger(cfg)).cuda()
        else:
            encoder = torch.nn.DataParallel(Encoder(cfg))
            decoder = torch.nn.DataParallel(Decoder(cfg))
            refiner = torch.nn.DataParallel(Refiner(cfg))
            merger = torch.nn.DataParallel(Merger(cfg))

        print('[INFO] %s Loading weights from = %s' % (dt.now(), cfg.CONST.WEIGHTS))

        if (torch.cuda.is_available()):
            encoder.load_state_dict(torch.load(cfg.CONST.WEIGHTS)['encoder_state_dict'])
            encoder.to(torch.device("cuda"))
            decoder.load_state_dict(torch.load(cfg.CONST.WEIGHTS)['decoder_state_dict'])
            decoder.to(torch.device("cuda"))
            merger.load_state_dict(torch.load(cfg.CONST.WEIGHTS)['merger_state_dict'])
            merger.to(torch.device("cuda"))
            refiner.load_state_dict(torch.load(cfg.CONST.WEIGHTS)['refiner_state_dict'])
            refiner.to(torch.device("cuda"))
        else:
            encoder.load_state_dict(torch.load(cfg.CONST.WEIGHTS, map_location=('cpu'))['encoder_state_dict'])
            decoder.load_state_dict(torch.load(cfg.CONST.WEIGHTS, map_location=('cpu'))['decoder_state_dict'])
            merger.load_state_dict(torch.load(cfg.CONST.WEIGHTS, map_location=('cpu'))['merger_state_dict'])
            refiner.load_state_dict(torch.load(cfg.CONST.WEIGHTS, map_location=('cpu'))['refiner_state_dict'])

    print('[INFO] %s Switching model to evaluate mode.' % (dt.now()))

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    merger.eval()
    refiner.eval()

    for sample_idx, (rendering_images) in enumerate(test_data_loader):
        # Get data from data loader
        print('\n')
        print('[INFO] %s Running Encoder.' % (dt.now()))
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        image_features = encoder(rendering_images)

        print('\n')
        print('[INFO] %s Running Decoder.' % (dt.now()))
        raw_features, generated_volume = decoder(image_features)

        if cfg.NETWORK.USE_MERGER:
            print('\n')
            print('[INFO] %s Running Merger.' % (dt.now()))
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)
        if cfg.NETWORK.USE_REFINER:
            print('\n')
            print('[INFO] %s Running Refiner.' % (dt.now()))
            generated_volume = refiner(generated_volume)
            print('\n')

        # Append generated volumes to TensorBoard
        print('[INFO] %s Generating volume.' % (dt.now()))

        gv = generated_volume.detach().cpu().numpy()

        # Volume Visualization
        if(plot):
            print('[INFO] %s Plotting volume.' % (dt.now()))

            utils.binvox_visualization.get_volume_views(gv)

        print('[INFO] %s Saving vox file in .\\vox_output\\.' % (dt.now()))

        vox = Vox.from_dense(gv.squeeze().__ge__(0.5))

        VoxWriter('.\\vox_output\\Vox.vox', vox).write()
        print('[INFO] %s Vox file saved correctly.' % (dt.now()))

def get_args_from_command_line():
    print(torch.__version__)
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file (default Pix2Vox-A-ShapeNet.pth)', default='.\\weights\\Pix2Vox-A-ShapeNet.pth')
    parser.add_argument('--image_folder_path', dest='image_folder_path', help='Image folder file path', default=None)
    parser.add_argument('--plot', dest='plot', help='Vox Preview [True/False] (default False)', default='False')
    parser.add_argument('--extension', dest='extension', help='Images file Extension [png/jpeg/jpg](default png)', default='png')
    try:
        args = parser.parse_args()
    except:
        raise Exception('[FATAL] %s Please insert the correct argumentm use -h' % (dt.now()))
    return args

if __name__ == '__main__':

    args = get_args_from_command_line()

    if sys.version_info < (3, 0):
        raise Exception('[FATAL] %s Please follow the installation instruction on "https://github.com/hzxie/Pix2Vox"' % (dt.now()))

    if (torch.cuda.is_available()):
        print('[INFO] %s Found CUDA device : %s' % (dt.now(), torch.cuda.get_device_name(torch.cuda.current_device())))
        cfg.CONST.DEVICE = torch.cuda.current_device()
        time.sleep(1)
    else:
        print('[INFO] %s No CUDA device found, using cpu instead.' % (dt.now()))
        time.sleep(1)

    print('[INFO] %s Getting arguments: [weight = %s] [image_folder_path = %s] [plot = %s] [extension = %s]' % (
        dt.now(), args.weights, args.image_folder_path, args.plot, args.extension))
    if (args.image_folder_path is None):
        raise Exception('[FATAL] %s Please specify the input image folder path.' % (dt.now()))

    if (not os.path.exists(args.image_folder_path)):
        raise Exception('[FATAL] %s Please specify a correct input image folder path.' % (dt.now()))

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    if (args.plot == 'False'):
        args.plot = False
    else:
        if (args.plot == 'True'):
            args.plot = True
        else:
            raise Exception('[FATAL] %s Please specify the plot args [True/False].' % (dt.now()))

    if (args.extension != 'png' and args.extension != 'jpg' and args.extension != 'jpeg'):
        raise Exception('[FATAL] %s Please insert a valid image extension [png/jpg/jpeg].' % (dt.now()))

    if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
        test_net(cfg, path=args.image_folder_path, plot=args.plot, extension=args.extension)
    else:
        raise Exception('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))



