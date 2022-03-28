import os
import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image


class DigitDataset(Dataset):
    def __init__(self, txt_path, images_path, transform=True, slices=6, x=52, y=52):
        self.transform = transform
        self.imgs_path = []
        self.words = []
        self.x = x
        self.y = y
        self.slices = slices
        f = open(txt_path, 'r')
        lines = f.readlines()
        is_first = True
        nums = []
        labels = []
        for line in lines:
            line = line.rstrip()

            if line.endswith('.jpg'):
                if is_first is True:
                    is_first = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line
                path = os.path.join(images_path, path)
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                if len(line) == 1:
                    num = int(line[0])
                    nums.append(num)
                else:
                    label = [float(x) for x in line]
                    labels.append(label)
        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        height, width = img.size
        h_resize, w_resize = 416, 416
        if self.transform:
            self.transform = transforms.Compose([transforms.Resize([h_resize, w_resize]),
                                                 transforms.ToTensor()])
            img = self.transform(img)
        labels = self.words[index]
        annotations = np.zeros((0, 5))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))
            # if label[0] > width - 1 or label[1] > height - 1:
            #     continue
            # if label[2] + label[0] > width - 1:
            #     label[2] = width - 1 - label[0]
            # if label[3] + label[1] > height - 1:
            #     label[3] = height - 1 - label[1]

            annotation[0, 0] = label[0]
            annotation[0, 1] = (label[1] + label[3] / 2) / width  # x
            annotation[0, 2] = (label[2] + label[4] / 2) / height  # y
            annotation[0, 3] = label[3] / width  # w
            annotation[0, 4] = label[4] / height  # h

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        mask = torch.zeros([self.slices, self.y, self.x])
        for label in target:
            cls = label[0]
            grid_x = int(label[1] * self.x)
            grid_y = int(label[2] * self.y)
            center_x = label[1] * self.x - grid_x
            center_y = label[2] * self.y - grid_y
            width = label[3]
            height = label[4]
            mask[0][grid_y][grid_x] = 1.0
            mask[1][grid_y][grid_x] = cls
            mask[-4][grid_y][grid_x] = center_x
            mask[-3][grid_y][grid_x] = center_y
            mask[-2][grid_y][grid_x] = width
            mask[-1][grid_y][grid_x] = height
        return img, mask
