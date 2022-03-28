import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DetectNet

imgs_path = r"D:\tmp\out\test_images"
imgs_name = os.listdir(imgs_path)
grid_yszie, grid_xsize = 8, 8
net = DetectNet()
net.load_state_dict(
    torch.load("./output/model.pth", map_location="cpu"))
with torch.no_grad():
    for img_name in imgs_name:
        img = Image.open(os.path.join(imgs_path, img_name))
        img_arr = np.array(img)
        transform = transforms.ToTensor()
        img = transform(img)
        img = img.unsqueeze(0)
        out = net(img)
        obj_pred = out[0].squeeze()
        cls_pred = out[1].squeeze(0)
        coord_pred = out[2].squeeze(0)
        obj = torch.sigmoid(obj_pred).cpu().numpy()
        center_coord = torch.sigmoid(coord_pred[0:-2]).cpu().numpy()
        loc = torch.sigmoid(coord_pred[-2:]).cpu().numpy()
        cls = torch.softmax(cls_pred, dim=0).cpu().numpy()
        cls = np.argmax(cls, axis=0)
        obj = (obj > 0.5).astype(np.uint8)
        idxs = np.argwhere(obj == 1)
        for idx in idxs:
            y = idx[0]
            x = idx[1]
            x_center = int((x + center_coord[0][y][x]) * grid_xsize)
            y_center = int((y + center_coord[1][y][x]) * grid_yszie)
            width = int(loc[0][y][x] * img_arr.shape[1])
            height = int(loc[1][y][x] * img_arr.shape[0])
            x_start = x_center - int(width / 2)
            y_start = y_center - int(height / 2)
            x_end = x_center + int(width / 2)
            y_end = y_center + int(height / 2)
            character_x = x_start - 3 if x_start - 3 >= 0 else 0
            character_y = y_start - 3 if y_start - 3 >= 0 else 0
            img_arr = cv2.putText(img_arr, str(cls[y, x]), (character_x, character_y), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 0, 255), 1, cv2.LINE_AA)
            img_arr = cv2.rectangle(img_arr, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.imshow("Image", img_arr)
        cv2.waitKey(0)
