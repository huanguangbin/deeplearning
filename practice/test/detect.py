import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from net.model import DetectNet

imgs_path = "../data/test_images"
imgs_name = os.listdir(imgs_path)
grid_yszie, grid_xsize = 8, 8
net = DetectNet()
net.load_state_dict(
    torch.load("./out/epoch_20.pth", map_location="cpu"))
with torch.no_grad():
    for img_name in imgs_name:
        img = Image.open(os.path.join(imgs_path, img_name))
        img_arr = np.array(img)
        transform = transforms.ToTensor()
        img = transform(img)
        img = img.unsqueeze(0)
        out = net(img)
        out = out.squeeze(0)
        out = torch.sigmoid(out).cpu().numpy()
        cls = (out[0] > 0.5).astype(np.uint8)
        idxs = np.argwhere(cls == 1)
        for idx in idxs:
            y = idx[0]
            x = idx[1]
            x_center = int((x + out[1][y][x]) * grid_xsize)
            y_center = int((y + out[2][y][x]) * grid_yszie)
            width = int(out[3][y][x] * img_arr.shape[1])
            height = int(out[4][y][x] * img_arr.shape[0])
            x_start = x_center-int(width/2)
            y_start = y_center-int(height/2)
            x_end = x_center+int(width/2)
            y_end = y_center+int(height/2)
            img_arr = cv2.rectangle(img_arr, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
        cv2.imshow("Image", img_arr)
        cv2.waitKey(0)
