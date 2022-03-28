# -*- coding:utf-8 -*-
import os
import copy
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


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
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

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


class ElasticDeformation:
    def __init__(self, random_state, spline_order=3, alpha=2000, sigma=50, execution_probability=1, **kwargs):
        """
        Args-
        random_state(np.random.mtrand): Random state object to decide on transformation execution.
        order(int), optional: The order for interpolation (use 0 for labeled images).
        alpha(int, optional): Scaling factor for deformations.
        sigma(float, optional): Std deviation for the Gaussian filter.
        execution_probability(float, optional): Probability of transformation execution.

        Kwargs-
        image_type(str): Image type 'raw', 'label' of the input image to use suitable 'order'.
        """
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.spline_order = spline_order
        self.execution_probability = execution_probability

    def __call__(self, input_array):
        if self.random_state.uniform() <= self.execution_probability:
            z_dim, y_dim, x_dim = input_array.shape
            dz, dy, dx = [gaussian_filter(self.random_state.randn(*input_array.shape),
                                          self.sigma, mode="constant") * self.alpha for _ in range(3)]
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx
            return map_coordinates(input_array, indices, order=self.spline_order, mode='constant')

        return input_array


class ElasticDeformation_Unet:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=2000, sigma=50, execution_probability=1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]
            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape
            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="constant") * self.alpha
            else:
                dz = np.zeros_like(m)
            dy, dx = [
                gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="constant") * self.alpha for
                _ in range(2)]
            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx
            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode="constant")
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode="constant") for c in
                            m]  # 'reflect'
                return np.stack(channels, axis=0)
        return m


if __name__ == '__main__':
    img_path = r"D:\Dataset\Intestinal\train_17groups\3D_imgs_npy_714"
    msk_path = r"D:\Dataset\Intestinal\train_17groups\3D_masks_npy_714"
    img_list = os.listdir(img_path)
    for img_name in img_list:
        img = np.load(os.path.join(img_path, img_name))
        msk = np.load(os.path.join(msk_path, img_name))
        img = img.squeeze(3)
        msk = msk.squeeze(3)
        image = copy.deepcopy(img)
        mask = copy.deepcopy(msk)
        randint = np.random.randint(0, 10000)
        random_state = np.random.RandomState(90)
        print(f"randint =======>{randint}")
        ela_img = ElasticDeformation(random_state, spline_order=3)
        img_out = ela_img(img)
        ela_msk = ElasticDeformation(random_state, spline_order=0)
        msk_out = ela_img(msk)

        for idx, _ in enumerate(img_out):
            mask[idx, -1, :] = 150
            trans = np.concatenate((img_out[idx], msk_out[idx] * 255), axis=1)
            ori = np.concatenate((image[idx], mask[idx] * 255), axis=1)
            res = np.concatenate((ori, trans), axis=0)
            cv2.namedWindow("img", 0)
            cv2.resizeWindow("img", 640, 640)
            cv2.imshow("img", res)
            cv2.waitKey(0)
