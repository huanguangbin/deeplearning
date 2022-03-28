import torch
import numpy as np

def iou_score(output, target, threshold):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output = output > threshold
    intersection = (output.astype(np.uint8) * target).sum()
    union = output.astype(np.uint8).sum() + target.sum() - intersection
    return intersection / (union + smooth)


def dice_coef(output, target, threshold):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
        output = output > threshold
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output.astype(np.uint8) * target).sum()
    return (2. * intersection) / (output.astype(np.uint8).sum() + target.sum() + smooth)
