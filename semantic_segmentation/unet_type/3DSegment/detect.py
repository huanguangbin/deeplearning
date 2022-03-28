import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
import torch
import unet3d
import joblib
import SimpleITK as sitk
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='Intestinal16_resize_unet3d_woDS',
                        help='model name')
    parser.add_argument('--arch_name', default='unet3d',
                        help='name of model architecture')
    parser.add_argument('--imgs_path', default=r'D:\Dataset\Intestinal\for test\imgs',
                        help='.nii.gz form')
    parser.add_argument('--masks_path', default=r'D:\Dataset\Intestinal\for test\msks',
                        help='.nii.gz form')
    parser.add_argument('--output_path', default=r'D:\Dataset\Intestinal\for test\infs',
                        help='.nii.gz form')
    parser.add_argument('--block_size', default=(32, 160),
                        help='')
    parser.add_argument('--threshold', default=0.5,type=int,
                        help='')
    parser.add_argument('--left_height', default=480,type=int,
                        help='')

    args = parser.parse_args()

    return args

def divide_img(img,dis_size):
    #   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h = img.shape[1]
    w = img.shape[2]
    n = h//dis_size
    m = w// dis_size  # 一行可切图像个数
    splitList=[]
    leftoverH = h % dis_size
    leftoverW = w % dis_size
    #print('h={},w={},n={}'.format(h, w, n))
    #num = 0
    for i in range(n):
        for j in range(m):
            #num += 1
            #print('i,j={},{}'.format(i, j))
            sub = img[:,dis_size * i:dis_size * (i + 1), dis_size * j:dis_size * (j + 1)]
            splitList.append(sub)
           # cv2.imwrite(save_path + img_name[:-4] + '_{}.png'.format(num), sub)
        if leftoverW !=0:
            #num += 1
            #print('-----')
            sub = img[:,dis_size * i:dis_size * (i + 1), -dis_size:]
            splitList.append(sub)
            #cv2.imwrite(save_path + img_name[:-4] + '_{}.png'.format(num), sub)
    if leftoverH !=0:
        for j in range(m):
            #num += 1
            #print('++++++')
            sub = img[:,-dis_size:,dis_size * j:dis_size * (j + 1)]
            splitList.append(sub)
            #cv2.imwrite(save_path + img_name[:-4] + '_{}.png'.format(num), sub)
        if leftoverW !=0:
            #num += 1
            #print('-----')
            sub = img[:,-dis_size :, -dis_size:]
            splitList.append(sub)
    return splitList

def cut_3Dimg(img,dis_block_num):
    img_block_list=[]
    for col in range(0, img.shape[0]-img.shape[0]%dis_block_num, dis_block_num):
        img_block=img[col:col+dis_block_num,:,:]
        img_block_list.append(img_block)
    if img.shape[0]%dis_block_num !=0:#但原图不能被切块层数整除时
        img_block = img[-dis_block_num:, :, :]
        img_block_list.append(img_block)
    return img_block_list

def stack_3Dimg(img,dis_block_num):
    pass

def crop_height(img,h):
    height,width = img[0].shape
    assert height>h
    return img[:, :h, :]

def image_compose(out_patches,img,dis_size,height):
    h = img.shape[1]
    w = img.shape[2]
    n = height//dis_size
    m = w// dis_size  # 一行可切图像个数
    leftoverH = height % dis_size
    leftoverW = w % dis_size
    to_image = np.zeros_like(img,dtype=np.uint8)  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    num=0
    print("Start Composing...")
    for y in range(n):
        for x in range(m):
            to_image[:,dis_size * y:dis_size * (y + 1),dis_size * x:dis_size * (x + 1)]=out_patches[num]
            num=num+1
        if leftoverW!=0:
            to_image[:,dis_size * y:dis_size * (y + 1), -dis_size:] = out_patches[num]
            num=num+1
    if leftoverH !=0:
        for j in range(m):
            to_image[:,height-dis_size:height,dis_size * j:dis_size * (j + 1)]=out_patches[num]
            num += 1
        if leftoverW != 0:
            to_image[:,height-dis_size:height, -dis_size:] = out_patches[num]
    return to_image

def inference_one_block(model, image_block,threshold, device):#image_block[32,160,160]
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with torch.no_grad():
            # img_block=torch.unsqueeze(image_block,0)
            # img_block=image_block
            img_block = torch.tensor(image_block)
            input=torch.unsqueeze(img_block,0)#[1,32,160,160]
            input = torch.unsqueeze(input, 0)#[1,1,32,160,160]
            input = input.to(device=device, dtype=torch.float32)
            output = model(input)
            output = torch.sigmoid(output).data.cpu().numpy()
            out = np.zeros_like(output,dtype=np.uint8)
            out[np.where(output >= threshold)] = 1
            out = np.squeeze(out)
    return out

def main():
    val_args = parse_args()
    args = joblib.load('models/%s/args.pkl' %val_args.name)
    if not os.path.exists(val_args.output_path):
        os.makedirs(val_args.output_path)
        print('输出目录创建成功')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=> creating model %s" %args.arch)
    model = unet3d.__dict__[args.arch](args)

    model = model.to(device=device)

    # Data loading code
    # img_paths = glob(r'D:\Dataset\LiverDataset\for train\3D_imgs_npy\*')#D:\Dataset\2-MICCAI_BraTS_2018\reprogress\testImage\
    # mask_paths = glob(r'D:\Dataset\LiverDataset\for train\3D_masks_npy\*')#D:\Dataset\2-MICCAI_BraTS_2018\reprogress\testMask\

    # val_img_paths = img_paths
    # val_mask_paths = mask_paths

    #model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model_dic = torch.load('models/%s/model.pth' %args.name, map_location='cpu')
    model.load_state_dict({k.replace("module.", ""): v for k, v in model_dic.items()})
    for niifiles in tqdm(os.listdir(val_args.imgs_path)):
        image_src = sitk.ReadImage(os.path.join(val_args.imgs_path,niifiles), sitk.sitkUInt8)
        image_array = sitk.GetArrayFromImage(image_src)
        image_array=image_array/255
        crop_image_array=crop_height(image_array, val_args.left_height)
        inference_list=[]
        split_imgs_list=divide_img(crop_image_array,val_args.block_size[1])
        # count = 0
        for split_imgs in tqdm(split_imgs_list):
            # #test
            # count += 1
            # split_sitk = sitk.GetImageFromArray(split_imgs)
            # sitk.WriteImage(split_sitk, os.path.join(r'D:\Dataset\LiverDataset\for train\test\800_160_160\imgs',
            #                                      niifiles[:-7] +'_'+ str(count) + '.nii.gz'))
            # #test
            imgs_block_list=cut_3Dimg(split_imgs,val_args.block_size[0])
            long_block=np.zeros([image_array.shape[0],val_args.block_size[1],val_args.block_size[1]])#先把32*160*160的小块拼成800*160*160的长块
            for num,imgs_block in tqdm(enumerate(imgs_block_list)):
                out=inference_one_block(model, imgs_block, val_args.threshold,device)
                if val_args.block_size[0]*(num+1)>image_array.shape[0]: #如果碰到重叠块，向前叠加
                    long_block[-val_args.block_size[0]:, :, :] = out
                else:
                    long_block[val_args.block_size[0]*num:val_args.block_size[0]*(num+1),:,:]=out
            inference_list.append(long_block)
            # #test
            # mytest=sitk.GetImageFromArray(long_block)
            # sitk.WriteImage(mytest, os.path.join(r'D:\Dataset\LiverDataset\for train\test\800_160_160\labels', niifiles[:-7] +'_'+str(count)+ '_inf.nii.gz'))
            # #test

        inference_array=image_compose(inference_list, image_array, val_args.block_size[1],val_args.left_height)
        for num,patch in tqdm(enumerate(inference_array)):
            cv2.imwrite(os.path.join(val_args.output_path,str(num)+'.png'),patch)
        #inference_sitk = sitk.GetImageFromArray(inference_array)
        #sitk.WriteImage(inference_sitk, os.path.join(val_args.output_path,niifiles[:-7]+'_inf.nii.gz'))
if __name__=="__main__":
    main()