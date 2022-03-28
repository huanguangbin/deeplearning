import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import unet3d
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
import joblib
import SimpleITK as sitk

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='',
                        help='path of model')
    parser.add_argument('--arch_name', default='unet3d',
                        help='name of model architecture')
    parser.add_argument('--imgs_path', default='',
                        help='.npy form')
    parser.add_argument('--masks_path', default='',
                        help='.npy form')
    parser.add_argument('--output_path', default='',
                        help='.nii.gz form')
    parser.add_argument('--batch_size', default=1,type=int,
                        help='')
    parser.add_argument('--mode', default=None,
                        help='')
    args = parser.parse_args()

    return args

def inference_one(net, image, device):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        output = torch.sigmoid(output).data.cpu().numpy()

def test():
    val_args = parse_args()
    model = unet3d.__dict__[val_args.arch](val_args)
    model = model.cuda()
    val_img_paths = glob(os.path.join(val_args.imgs_path,'*'))
    val_mask_paths = glob(os.path.join(val_args.masks_path,'*'))
    model.load_state_dict(torch.load(val_args.model))
    model.eval()
    val_dataset = Dataset(val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if not os.path.exists(val_args.output_path):
        os.mkdir(val_args.output_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    with torch.no_grad():
        startFlag = 1
        for mynum, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            output = model(input)
            output = torch.sigmoid(output).data.cpu().numpy()
            target = target.data.cpu().numpy()
            img_paths = val_img_paths[val_args.batch_size * mynum:val_args.batch_size * (mynum + 1)]

def detect():
    val_args = parse_args()
    model = unet3d.__dict__[val_args.arch](val_args)
    model = model.cuda()
    val_img_paths = glob(os.path.join(val_args.imgs_path,'*'))
    val_mask_paths = glob(os.path.join(val_args.masks_path,'*'))
    model.load_state_dict(torch.load(val_args.model))
    model.eval()
    val_dataset = Dataset(val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if not os.path.exists(val_args.output_path):
        os.mkdir(val_args.output_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    with torch.no_grad():
        startFlag = 1
        for mynum, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            output = model(input)
            output = torch.sigmoid(output).data.cpu().numpy()
            target = target.data.cpu().numpy()
            img_paths = val_img_paths[val_args.batch_size * mynum:val_args.batch_size * (mynum + 1)]








def main():
    args=parse_args()
    if args.mode=='detect':
        detect()
    elif args.mode=='test':
        test()