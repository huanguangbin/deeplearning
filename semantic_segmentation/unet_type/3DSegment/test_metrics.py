import os
import cv2
import torch
import numpy as np
from collections import OrderedDict
from unet3d import UNet3D
from vnet3d import VNet
from metrics import iou_score, dice_coef
from PIL import Image


def test_metrics(model_path="./output/v_norm17/v_Norm17.pth"):
    npy_imgs_path = "/home/test/hgb/data/Intestinal/test_set/imgs"
    npy_msks_path = "/home/test/hgb/data/Intestinal/test_set/msks"
    npy_names = os.listdir(npy_imgs_path)
    #model = UNet3D()
    model = VNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        OrderedDict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()}))
    model.to(device=device)
    model.eval()
    Dice = 0.
    Iou = 0.
    with torch.no_grad():
        for npy_name in npy_names:
            npimage = np.load(os.path.join(npy_imgs_path, npy_name))
            #image = npimage.squeeze(3).astype(np.uint8)
            npmask = np.load(os.path.join(npy_msks_path, npy_name))
            npimage = npimage.transpose((3, 0, 1, 2))
            npmask = npmask.transpose((3, 0, 1, 2))
            npimage = npimage.astype("float32")
            npmask = npmask.astype("float32")
            npimage = npimage / 255
            npimage = (npimage - 0.3383) / 0.1146
            npimage = torch.tensor(npimage)
            npimage = npimage.unsqueeze(0).to(device=device)
            out = model(npimage)
            iou = iou_score(out, npmask, 0.5)
            dice = dice_coef(out, npmask, 0.5)
            Iou += iou
            Dice += dice
            print(f"iou:{iou} dice:{dice}")
            # out = out.squeeze(0)
            # out = out.squeeze(0)
            # out[out >= 0.5] = 255
            # out[out < 0.5] = 0
            # out = out.cpu().numpy().astype(np.uint8)
            # res = np.full((1280, 1280), 255)
            # for idx, img in enumerate(out):
            #     Y = (idx // 4)*160
            #     X = (idx % 4)*320
            #     # axis = 0 if (idx + 1) % 4 == 0 else 1
            #     tmp = np.concatenate((image[idx], img), axis=1)
            #     res[Y:Y + 160, X:X + 320] = tmp

            # res = Image.fromarray(res.astype(np.uint8))
            # res.show()
    print(f"Iou:{Iou / len(npy_names)} Dice:{Dice / len(npy_names)}")


if __name__ == "__main__":
    test_metrics()
