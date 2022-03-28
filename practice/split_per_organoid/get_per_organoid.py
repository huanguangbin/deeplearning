# coding=utf-8
import os
import random

import numpy as np
import cv2
import time
from tqdm import tqdm
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1])  # , borderMode=cv2.BORDER_REFLECT_101

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)

    return imageC


# elastic_transform
def split_per_organoid_trans(img, enlarge_size=0, resize_size=(28, 28), trans=True):
    """
    :param img: 输入图像
    :param enlarge_size: 边界扩张像素值
    :param resize_size:
    :param trans: 弹性+仿射变换
    :return:
    """
    imgs_dict = dict()
    imgs_dict["crop"] = []
    imgs_dict["elastic"] = []
    assert len(img.shape) == 2, "The image should be a gray image"
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.squeeze(1)
        min_x = max(points[:, 0].min() - enlarge_size, 0)
        max_x = min(points[:, 0].max() + enlarge_size, img.shape[1])
        min_y = max(points[:, 1].min() - enlarge_size, 0)
        max_y = min(points[:, 1].max() + enlarge_size, img.shape[0])
        crop_img = img[min_y:max_y, min_x:max_x]
        # cv2.imshow("tmp", crop_img)
        # cv2.waitKey()
        crop_img = cv2.resize(crop_img, resize_size, interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("tmp2", crop_img)
        # cv2.waitKey()
        crop_img[crop_img > 50] = 255
        crop_img[crop_img < 50] = 0
        imgs_dict["crop"].append(crop_img)
        if trans:
            alpha_coff = crop_img.shape[1] * random.uniform(0.02, 0.2)
            sigma_coff = crop_img.shape[1] * random.uniform(0.01, 0.1)
            affine_coff = crop_img.shape[1] * random.uniform(0.01, 0.1)
            affine_img = elastic_transform(crop_img, alpha_coff, sigma_coff, affine_coff)
            affine_img[affine_img > 50] = 255
            affine_img[affine_img < 50] = 0
            imgs_dict["elastic"].append(affine_img)
            # img_show = np.concatenate((crop_img, affine_img), axis=1)
            # cv2.imshow("tmp", img_show)
            # cv2.waitKey()
    return imgs_dict


# all resize
def split_per_organoid_1(img, enlarge_size=0, resize_size=(28, 28), affine=True):
    imgs_dict = dict()
    imgs_dict["crop"] = []
    imgs_dict["affine"] = []
    assert len(img.shape) == 2, "The image should be a gray image"
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.squeeze(1)
        min_x = max(points[:, 0].min() - enlarge_size, 0)
        max_x = min(points[:, 0].max() + enlarge_size, img.shape[1])
        min_y = max(points[:, 1].min() - enlarge_size, 0)
        max_y = min(points[:, 1].max() + enlarge_size, img.shape[0])
        crop_img = img[min_y:max_y, min_x:max_x]
        # cv2.imshow("tmp", crop_img)
        # cv2.waitKey()
        crop_img = cv2.resize(crop_img, resize_size, interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("tmp2", crop_img)
        # cv2.waitKey()
        crop_img[crop_img > 50] = 255
        crop_img[crop_img < 50] = 0
        imgs_dict["crop"].append(crop_img)
        if affine:
            rows, cols = resize_size
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
            pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
            M = cv2.getAffineTransform(pts1, pts2)
            affine_img = cv2.warpAffine(crop_img, M, (cols, rows))
            affine_img[affine_img > 50] = 255
            affine_img[affine_img < 50] = 0
            imgs_dict["affine"].append(affine_img)
    return imgs_dict


def split_per_organoid(img, enlarge_size=0, resize_size=(28, 28), affine=True, scale_threshold=0.2):
    imgs_dict = dict()
    imgs_dict["crop"] = []
    imgs_dict["affine"] = []
    assert len(img.shape) == 2, "The image should be a gray image"
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.squeeze(1)
        min_x = max(points[:, 0].min() - enlarge_size, 0)
        max_x = min(points[:, 0].max() + enlarge_size, img.shape[1])
        min_y = max(points[:, 1].min() - enlarge_size, 0)
        max_y = min(points[:, 1].max() + enlarge_size, img.shape[0])
        crop_img = img[min_y:max_y, min_x:max_x]
        # cv2.imshow("tmp", crop_img)
        # cv2.waitKey()
        h, w = crop_img.shape
        if random.random() < scale_threshold:
            interpolate_size = (int(resize_size[0] * (w / h)), resize_size[1]) if h > w else (
                resize_size[0], int(resize_size[1] * (h / w)))
        else:
            interpolate_size = (w, h)
        crop_img = cv2.resize(crop_img, interpolate_size, interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("tmp2", crop_img)
        # cv2.waitKey()
        height_pad = max(resize_size[0] - interpolate_size[1], w - h)
        width_pad = max(resize_size[1] - interpolate_size[0], h - w)
        if height_pad < 0:
            height_pad = 0
        if width_pad < 0:
            width_pad = 0
        crop_img = np.pad(crop_img, (
            (int(height_pad / 2), height_pad - int(height_pad / 2)),
            (int(width_pad / 2), width_pad - int(width_pad / 2))),
                          'constant', constant_values=0)
        crop_img[crop_img > 50] = 255
        crop_img[crop_img < 50] = 0
        imgs_dict["crop"].append(crop_img)
        if affine:
            rows, cols = resize_size
            pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
            pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
            M = cv2.getAffineTransform(pts1, pts2)
            affine_img = cv2.warpAffine(crop_img, M, (cols, rows))
            affine_img[affine_img > 50] = 255
            affine_img[affine_img < 50] = 0
            imgs_dict["affine"].append(affine_img)
    return imgs_dict


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def main():
    files_dir = r"D:\tmp\data_pix2pix\GAN_G\masks"
    save_dir = r"D:\tmp\data_pix2pix\GAN_G\crop_elastic\big"
    save_category = ["elastic"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("save directory created!")
    files_name = os.listdir(files_dir)
    for file_name in tqdm(files_name):
        preffix, suffix = file_name.split('.')
        file_path = os.path.join(files_dir, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # results = split_per_organoid(img, enlarge_size=5, resize_size=(20, 20))
        results = split_per_organoid_trans(img, enlarge_size=5, resize_size=(64, 64))
        for category in results:
            if category in save_category and len(results[category]) > 0:
                for idx, per_img in enumerate(results[category]):
                    save_file_name = preffix + category + '_' + str(idx) + '.' + suffix
                    save_path = os.path.join(save_dir, save_file_name)
                    cv2.imwrite(save_path, per_img)


if __name__ == "__main__":
    main()
