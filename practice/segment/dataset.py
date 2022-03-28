from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset


class SemDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.imgs_names = os.listdir(imgs_dir)

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, index):
        fn = self.imgs_names[index]
        img = Image.open(os.path.join(self.imgs_dir, fn))
        img_nd = np.array(img)
        label = Image.open(os.path.join(self.masks_dir, fn))
        label_nd = np.array(label)
        if len(label_nd.shape) == 2:
            label_nd = np.expand_dims(label_nd, axis=2)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # img_nd=img_nd.transpose((2, 0, 1))
        label_nd = label_nd.transpose((2, 0, 1))
        if self.transform is not None:
            img = self.transform(img_nd)
        else:
            img = img_nd.transpose((2, 0, 1))
        return img, label_nd
