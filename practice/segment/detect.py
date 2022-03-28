import torch
import cv2
from torchvision import transforms
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from net import NetSem


def pred_img(model, img_path, transform=transforms.ToTensor()):
    model.eval()
    img = Image.open(img_path)
    img_nd = np.array(img)
    img_nd = transform(img_nd)
    img_nd = img_nd.unsqueeze(0)
    with torch.no_grad():
        output = model(img_nd)
        prob = F.softmax(output, dim=1)
        prob = prob.squeeze(0).cpu().numpy()
        pred = np.zeros([prob.shape[1], prob.shape[2]])
        for i, patch in enumerate(prob):
            pred[np.where(patch >= 0.5)] = i
        return pred.astype(np.uint8)


if __name__ == "__main__":
    color = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    model = NetSem()
    model.load_state_dict(torch.load('output/model.pth'))
    imgs = './data/test_imgs'
    show_img = False
    for img_name in tqdm(os.listdir(imgs)):
        img_path = os.path.join(imgs, img_name)
        img = cv2.imread(img_path)
        pred = pred_img(model, img_path)
        pred_color = np.zeros([pred.shape[0], pred.shape[1], 3])
        for i in range(pred.max() + 1):
            pred_color[np.where(pred == i)] = color[i]
        cv2.imwrite(os.path.join('./data/pred', img_name), pred_color.astype(np.uint8))
        if show_img:
            img_concat = np.concatenate((img, pred_color), axis=1)
            cv2.imshow("img", img_concat)
            cv2.waitKey()

