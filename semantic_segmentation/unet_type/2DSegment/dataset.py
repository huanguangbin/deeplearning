import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from augment import OctAug


class BasicDataset(Dataset):
    def __init__(self, imgs_paths, masks_paths, scale=1, aug=False):
        self.imgs_paths = imgs_paths
        self.masks_paths = masks_paths
        self.scale = scale
        self.aug = aug
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

    def __len__(self):
        return len(self.imgs_paths)

    @classmethod
    def preprocess(cls, pil_img, pil_msk, scale, aug=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        pil_msk = pil_msk.resize((newW, newH))

        img_nd = np.array(pil_img)
        msk_nd = np.array(pil_msk)/255 #the label max pixel is 255
        if aug:
            aug_ = OctAug(img_nd, msk_nd)
            img_nd, msk_nd = aug_.norm_aug()
        msk_nd = np.expand_dims(msk_nd, axis=2)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        msk_trans = msk_nd.transpose((2, 0, 1))
        return img_trans.astype(float), msk_trans.astype(np.uint8)

    def __getitem__(self, i):
        img_path = self.imgs_paths[i]
        mask_path = self.masks_paths[i]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        assert img.size == mask.size, \
            f'Image and mask {img_path} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask, self.scale, aug=self.aug)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
