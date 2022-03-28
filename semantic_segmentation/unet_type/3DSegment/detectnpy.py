import unet3d
import joblib
import SimpleITK as sitk
from tqdm import tqdm
import warnings
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import torch
def inference_one_block(model, image_block,threshold, device):#image_block[32,160,160]
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with torch.no_grad():
            # img_block=torch.unsqueeze(image_block,0)
            # img_block=image_block
            img_block = torch.tensor(image_block)
            input=torch.unsqueeze(img_block,0)#[1,32,160,160]
            input = torch.unsqueeze(input, 0)#[1,1,32,160,160]
            input = input.to(device=device, dtype=torch.float32)
            output = model(input)
            output = torch.sigmoid(output).data.cpu().numpy()
            out = np.zeros_like(output,dtype=np.uint8)
            out[np.where(output >= threshold)] = 1
            out = np.squeeze(out)
    return out
if __name__=="__main__":
    image_block=
