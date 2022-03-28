import copy

import SimpleITK as sitk
import numpy as np
import torch
import warnings
import argparse
from glob import glob
import os
import logging
from unet3d import UNet3D
# from vnet3d import VNet
from vnet1 import VNet
from metrics import dice_coef, iou_score
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./output/v_norm17/unet3d.pth",
                        # './output/VNet_attention/vnet_att.pth',
                        help='model name')
    parser.add_argument('--vol_data_path', default=r'/home/test/hgb/data/Intestinal/metrics_set/imgs',
                        help='.nii.gz form')
    parser.add_argument('--vol_mask_path', default=r'/home/test/hgb/data/Intestinal/metrics_set/masks',
                        help='.nii.gz form')
    parser.add_argument('--block_size', default=(32, 160, 160),
                        help='')
    parser.add_argument('--threshold', default=0.5, type=int,
                        help='')
    parser.add_argument('--left_height', default=480, type=int,
                        help='')
    parser.add_argument('--show_image', default=False, type=bool,
                        help='')
    parser.add_argument('--overlap_compose_rate', default=0.2, type=float,
                        help='between 0-1')
    parser.add_argument('--img_suffix', default="_ori.nii.gz", type=str,
                        help='')
    parser.add_argument('--msk_suffix', default="_seg.nii.gz", type=str,
                        help='')
    parser.add_argument('--max_pixel', default=99, type=float,
                        help='')
    parser.add_argument('--mean', default=0.3383, type=float,
                        help='')
    parser.add_argument('--std', default=0.0709, type=float,
                        help='')

    args = parser.parse_args()

    return args


def cal_loc__discard_version(coord, size):
    z_step, y_step, x_step = size
    block_list = []
    for z_coord in range(0, coord[0], z_step):
        if z_coord + z_step > coord[0] and coord[0] % z_coord != 0:
            for y_coord in range(0, coord[1], y_step):
                if y_coord + y_step > coord[1] and coord[1] % y_coord != 0:
                    for x_coord in range(0, coord[2], x_step):
                        if x_coord + x_step > coord[2] and coord[2] % x_coord != 0:
                            loc_z = coord[0] - z_step
                            loc_y = coord[1] - y_step
                            loc_x = coord[2] - x_step
                            block_list.append((loc_z, loc_y, loc_x))
                            break
                        loc_z = coord[0] - z_step
                        loc_y = coord[1] - y_step
                        loc_x = x_coord
                        block_list.append((loc_z, loc_y, loc_x))
                    break
                for x_coord in range(0, coord[2], x_step):
                    if x_coord + x_step > coord[2] and coord[2] % x_coord != 0:
                        loc_z = coord[0] - z_step
                        loc_y = y_coord
                        loc_x = coord[2] - x_step
                        block_list.append((loc_z, loc_y, loc_x))
                        break
                    loc_z = coord[0] - z_step
                    loc_y = y_coord
                    loc_x = x_coord
                    block_list.append((loc_z, loc_y, loc_x))
            break
        for y_coord in range(0, coord[1], y_step):
            if y_coord + y_step > coord[1] and coord[1] % y_coord != 0:
                for x_coord in range(0, coord[2], x_step):
                    if x_coord + x_step > coord[2] and coord[2] % x_coord != 0:
                        loc_z = z_coord
                        loc_y = coord[1] - y_step
                        loc_x = coord[2] - x_step
                        block_list.append((loc_z, loc_y, loc_x))
                        break
                    loc_z = z_coord
                    loc_y = coord[1] - y_step
                    loc_x = x_coord
                    block_list.append((loc_z, loc_y, loc_x))
                break
            for x_coord in range(0, coord[2], x_step):
                if x_coord + x_step > coord[2] and coord[2] % x_coord != 0:
                    loc_z = z_coord
                    loc_y = y_coord
                    loc_x = coord[2] - x_step
                    block_list.append((loc_z, loc_y, loc_x))
                    break
                loc_z = z_coord
                loc_y = y_coord
                loc_x = x_coord
                block_list.append((loc_z, loc_y, loc_x))
    return block_list


def cal_loc(coord, size, overlap_compose_rate):
    assert 0 <= overlap_compose_rate < 1
    z_size, y_size, x_size = size
    z_step, y_step, x_step = z_size, int(y_size * (1 - overlap_compose_rate)), int(x_size * (1 - overlap_compose_rate))
    block_list = []
    for z_coord in range(0, coord[0], z_step):
        if z_coord + z_size > coord[0]:
            z_coord = coord[0] - z_size
        for y_coord in range(0, coord[1], y_step):
            if y_coord + y_size > coord[1]:
                y_coord = coord[1] - y_size
            for x_coord in range(0, coord[2], x_step):
                if x_coord + x_size > coord[2]:
                    x_coord = coord[2] - x_size
                loc_z = z_coord
                loc_y = y_coord
                loc_x = x_coord
                block_list.append((loc_z, loc_y, loc_x))
                if x_coord + x_size == coord[2]:
                    break
            if y_coord + y_size == coord[1]:
                break
        if z_coord + z_size == coord[0]:
            break
    return block_list


def inference_one_block(model, image_block, threshold, device):  # image_block[32,160,160]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        input = torch.unsqueeze(image_block, 0)
        input = torch.unsqueeze(input, 0)
        input = input.to(device=device, dtype=torch.float32)
        output = model(input)
        output = torch.sigmoid(output).data.cpu().numpy()
        out = np.zeros_like(output, dtype=np.uint8)
        out[np.where(output >= threshold)] = 1
        out = np.squeeze(out)
    return out


def load_array(data_arry, msks_array):
    data_list = []
    mask_list = []
    for idx, patch in enumerate(msks_array):
        if patch.max() > 0:
            data_list.append(data_arry[idx])
            mask_list.append(patch)
    return np.array(data_list), np.array(mask_list)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"use device {device}")
    model = UNet3D()
    # model = VNet()
    model.load_state_dict(
        OrderedDict({k.replace('module.', ''): v for k, v in torch.load(args.model_path).items()}))
    model = model.to(device=device)
    model.eval()
    vol_data_list = os.listdir(args.vol_data_path)
    for vol_data in vol_data_list:
        iou = 0
        dice = 0
        vol_data_src = sitk.ReadImage(os.path.join(args.vol_data_path, vol_data), sitk.sitkUInt8)
        data_array = sitk.GetArrayFromImage(vol_data_src)
        vol_mask_name = vol_data.replace(args.img_suffix, args.msk_suffix)
        vol_mask_src = sitk.ReadImage(os.path.join(args.vol_mask_path, vol_mask_name), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(vol_mask_src)
        data_array, mask_array = load_array(data_array, mask_array)
        data_array1 = copy.deepcopy(data_array)
        data_array = data_array / args.max_pixel
        data_array = (data_array - args.mean) / args.std
        data_array = torch.tensor(data_array)
        inference_data = np.zeros_like(mask_array)
        loc_coord = (data_array.shape[0], args.left_height, data_array.shape[2])
        loc_list = cal_loc(loc_coord, args.block_size, args.overlap_compose_rate)
        with torch.no_grad():
            for loc in tqdm(loc_list):
                data_block = data_array[loc[0]:loc[0] + args.block_size[0], loc[1]:loc[1] + args.block_size[1],
                             loc[2]:loc[2] + args.block_size[2]]
                pred = inference_one_block(model, data_block, args.threshold, device)
                inference_data[loc[0]:loc[0] + args.block_size[0], loc[1]:loc[1] + args.block_size[1],
                loc[2]:loc[2] + args.block_size[2]] += pred
            inference_data[inference_data > 0] = 1
        for idx, patch in enumerate(inference_data):
            # show patches
            if args.show_image:
                vis_patch = patch * 255
                vis_patch[:, -1] = 150
                tmp0 = np.concatenate((vis_patch, mask_array[idx] * 255), axis=1)
                tmp1 = np.concatenate((data_array1[idx], data_array1[idx]), axis=1)
                tmp = np.concatenate((tmp1, tmp0), axis=0)
                res = Image.fromarray(tmp.astype(np.uint8))
                res.show()
            # show patches
            iou += iou_score(patch, mask_array[idx], args.threshold)
            dice += dice_coef(patch, mask_array[idx], args.threshold)
        iou /= inference_data.shape[0]
        dice /= inference_data.shape[0]
        logging.info(f"Iou = {iou} Dice = {dice}")


if __name__ == "__main__":
    main()
