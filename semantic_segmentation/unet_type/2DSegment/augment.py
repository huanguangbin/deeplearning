import cv2
import random
import numpy as np


class OctAug:
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask
        self.rotate_angle = random.uniform(-180, 180)
        self.shifit_limit = 20
        self.shifit_dx = round(random.uniform(-self.shifit_limit, self.shifit_limit))
        self.shifit_dy = round(random.uniform(-self.shifit_limit, self.shifit_limit))
        self.flip_axis = random.randint(-1, 1)
        self.gamma_coef = round(np.random.uniform(0.65, 1.5), 1)
        self.lum_ratio = round(np.random.uniform(0.7, 1.3), 1)

    # 仿射变换（角度自定）
    def rotate(self, img, msk, angle=0.):
        center = (img.shape[1] / 2, img.shape[0] / 2)
        size = (img.shape[1], img.shape[0])
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotate_img = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotate_msk = cv2.warpAffine(msk, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotate_img, rotate_msk

    # 浊化
    def blur(self, img, msk):
        blur_img = cv2.blur(img, (3, 3))
        return blur_img, msk

    # 裁剪（边缘模糊，不改变尺寸，边界自定）
    def shifit(self, img, msk, limit, dx, dy):
        if len(img.shape) == 2:
            height, width = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img_ = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_CONSTANT, value=0)
            img = img_[y1:y2, x1:x2]

            mask_ = cv2.copyMakeBorder(msk, limit + 1, limit + 1, limit + 1, limit + 1,
                                       borderType=cv2.BORDER_CONSTANT, value=0)
            mask = mask_[y1:y2, x1:x2]
        else:
            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img_ = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1,
                                      borderType=cv2.BORDER_REFLECT)
            img = img_[y1:y2, x1:x2, :]

            mask_ = cv2.copyMakeBorder(msk, limit + 1, limit + 1, limit + 1, limit + 1,
                                       borderType=cv2.BORDER_REFLECT)
            mask = mask_[y1:y2, x1:x2]

        return img, mask

    # 翻转（水平/垂直）
    def random_filp(self, img, msk, axis):
        img = cv2.flip(img, axis)
        mask = cv2.flip(msk, axis)
        return img, mask

    # 添加噪声
    def add_noise(self, img, msk, noise_percent=0.001):
        assert 0 < noise_percent < 1
        noise_num = int(img.shape[0] * img.shape[1] * noise_percent)
        for i in range(noise_num):
            noise_value = np.random.randint(0, 255)
            temp_x = np.random.randint(0, img.shape[0] - 1)
            temp_y = np.random.randint(0, img.shape[1] - 1)
            img[temp_x][temp_y] = noise_value
        return img, msk

    # 更改对比度
    def contrast_adjust(self, img, msk):
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        # cl1 = clahe.apply(img)
        pass
        return img, msk

    def aug(self):
        if round(np.random.uniform(0, 1), 1) <= 0.2:
            self.data, self.mask = self.rotate(self.data, self.mask, self.rotate_angle)
        if round(np.random.uniform(0, 1), 1) <= 0.2:
            self.data, self.mask = self.blur(self.data, self.mask)
        if round(np.random.uniform(0, 1), 1) <= 0.2:
            self.data, self.mask = self.shifit(self.data, self.mask, self.shifit_limit,
                                                                     self.shifit_dx, self.shifit_dy)
        if round(np.random.uniform(0, 1), 1) <= 0.2:
            self.data, self.mask = self.random_filp(self.data, self.mask, self.flip_axis)
        if round(np.random.uniform(0, 1), 1) <= 0.1:
            self.data, self.mask = self.add_noise(self.data, self.mask)
        if round(np.random.uniform(0, 1), 1) <= 0.2:
            self.data, self.mask = self.contrast_adjust(self.data, self.mask)
        return self.data, self.mask

    def norm_aug(self):
        self.aug()
        mean = 0.3383
        std = 0.0709
        max_voxel = 99.
        self.data = self.data / max_voxel
        if round(np.random.uniform(0, 1), 1) <= 0.2:  # 伽马变化
            self.data = np.power(self.data, self.gamma_coef)
        self.data = (self.data - mean) / std
        return self.data, self.mask
