import os
import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image


# class WiderFaceDataset(Dataset):
#     def __init__(self, txt_path, images_path, transform=True):
#         self.transform = transform
#         self.imgs_path = []
#         self.words = []
#         f = open(txt_path, 'r')
#         lines = f.readlines()
#         isFirst = True
#         nums = []
#         labels = []
#         for line in lines:
#             line = line.rstrip()
#
#             if line.endswith('.jpg'):
#                 if isFirst is True:
#                     isFirst = False
#                 else:
#                     labels_copy = labels.copy()
#                     self.words.append(labels_copy)
#                     labels.clear()
#                 path = line
#
#                 path = os.path.join(images_path, path)
#
#                 self.imgs_path.append(path)
#             else:
#                 line = line.split(' ')
#                 if len(line) == 1:
#                     num = int(line[0])
#                     nums.append(num)
#                 else:
#                     label = [float(x) for x in line]
#                     labels.append(label[:4])
#
#         self.words.append(labels)
#
#     def __len__(self):
#         return len(self.imgs_path)
#
#     def __getitem__(self, index):
#         # img = cv2.imread(self.imgs_path[index])#(678, 1024, 3)
#         img = Image.open(self.imgs_path[index])
#         width, height = img.size
#         if self.transform:
#             self.transform = transforms.Compose([transforms.Resize([416, 416]),
#                                                  transforms.ToTensor()])
#             img = self.transform(img)
#         # height, width, _ = img.shape
#         labels = self.words[index]
#         annotations = np.zeros((0, 4))
#         if len(labels) == 0:
#             return annotations
#         for idx, label in enumerate(labels):
#             annotation = np.zeros((1, 4))
#             if label[0] > width - 1 or label[1] > height - 1:
#                 continue
#             if label[2]+label[0] > width - 1:
#                 label[2] = width - 1 - label[0]
#             if label[3] + label[1] > height - 1:
#                 label[3] = height - 1 - label[1]
#
#             annotation[0, 0] = (label[0] + label[2] / 2) / width  # x1
#             annotation[0, 1] = (label[1] + label[3] / 2) / height  # y1
#             annotation[0, 2] = label[2] / width  # w
#             annotation[0, 3] = label[3] / height  # h
#
#             annotations = np.append(annotations, annotation, axis=0)
#         target = np.array(annotations)
#         return img, target


class WiderFaceDataset(Dataset):
    def __init__(self, txt_path, images_path, transform=True, slices=5, x=52, y=52):
        self.transform = transform
        self.imgs_path = []
        self.words = []
        self.x = x
        self.y = y
        self.slices = slices
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        nums = []
        labels = []
        for line in lines:
            line = line.rstrip()

            if line.endswith('.jpg'):
                if isFirst is True:
                    isFirst = False
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
                    labels.append(label[:4])

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        height, width = img.size
        h_resize,w_resize=416,416
        if self.transform:
            self.transform = transforms.Compose([transforms.Resize([h_resize, w_resize]),
                                                 transforms.ToTensor()])
            img = self.transform(img)
        labels = self.words[index]
        annotations = np.zeros((0, 4))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            # if label[0] > width - 1 or label[1] > height - 1:
            #     continue
            # if label[2] + label[0] > width - 1:
            #     label[2] = width - 1 - label[0]
            # if label[3] + label[1] > height - 1:
            #     label[3] = height - 1 - label[1]

            annotation[0, 0] = (label[0] + label[2] / 2) / width # x
            annotation[0, 1] = (label[1] + label[3] / 2) / height  # y
            annotation[0, 2] = label[2] / width  # w
            annotation[0, 3] = label[3] / height  # h

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        mask = torch.zeros([self.slices, self.y, self.x])
        for label in target:
            grid_x = int(label[0] * self.x)
            grid_y = int(label[1] * self.y)
            center_x = label[0] * self.x - grid_x
            center_y = label[1] * self.y - grid_y
            width = label[2]
            height = label[3]
            mask[0][grid_y][grid_x] = 1.0
            mask[1][grid_y][grid_x] = center_x
            mask[2][grid_y][grid_x] = center_y
            mask[3][grid_y][grid_x] = width
            mask[4][grid_y][grid_x] = height
        return img, mask
