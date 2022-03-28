import numpy as np
import torch.utils.data
from augment import OctAug


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False, val_mode=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.val_mode = val_mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        if self.val_mode or self.aug is False:
            npimage = npimage / 255.
            npimage = (npimage - 0.3383) / 0.1146
            # npimage = npimage / 99.
            # npimage = (npimage - 0.3383) / 0.0709
        elif self.aug:
            npimage = npimage.squeeze(3)
            npmask = npmask.squeeze(3)
            oct_aug = OctAug(npimage, npmask)
            npimage, npmask = oct_aug.norm_aug()
            npimage = np.expand_dims(npimage, 3)
            npmask = np.expand_dims(npmask, 3)

        npimage = npimage.transpose((3, 0, 1, 2))
        npmask = npmask.transpose((3, 0, 1, 2))
        npmask = npmask.astype("float32")
        npimage = npimage.astype("float32")
        return npimage, npmask
