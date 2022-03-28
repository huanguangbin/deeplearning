import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataloader.dataset import MyDataset
from net.model import DetectNet
from tqdm import tqdm


def train():
    txt_path = "../data/label.txt"
    imgs_path = "../data/train_images"
    save_path = "./out"
    epochs = 50
    val_percent = 0.2
    batch_size = 2
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    dataset = MyDataset(txt_path, imgs_path)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_data, val_data = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model = DetectNet()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    cls_loss_func = nn.BCEWithLogitsLoss()
    wh_loss_func = nn.SmoothL1Loss()
    loc_loss_func = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{epochs}', unit='img(s)') as pbar:
            for img, mask in train_loader:
                train_loss = 0
                model.train()
                img = img.cuda()
                out = model(img)
                mask = mask.cuda()
                train_loss += cls_loss_func(out[:, 0], mask[:, 0])
                train_loss += loc_loss_func(mask[:, 0] * out[:, 1], mask[:, 1])
                train_loss += loc_loss_func(mask[:, 0] * out[:, 2], mask[:, 2])
                train_loss += wh_loss_func(torch.sigmoid(mask[:, 0] * out[:, 3]), mask[:, 3])
                train_loss += wh_loss_func(torch.sigmoid(mask[:, 0] * out[:, 4]), mask[:, 4])
                pbar.set_postfix({'loss (batch)': train_loss.item()})
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                pbar.update(img.shape[0])
        scheduler.step()
        with torch.no_grad():
            with tqdm(total=len(val_data), desc=f'Validation round', unit='img(s)', leave=False) as pbar:
                val_loss = 0
                for img, mask in val_loader:
                    loss = 0
                    model.eval()
                    img = img.cuda()
                    out = model(img)
                    mask = mask.cuda()
                    loss += cls_loss_func(out[:, 0], mask[:, 0])
                    loss += loc_loss_func(torch.sigmoid(mask[:, 0] * out[:, 1]), mask[:, 1])
                    loss += loc_loss_func(torch.sigmoid(mask[:, 0] * out[:, 2]), mask[:, 2])
                    loss += wh_loss_func(mask[:, 0] * out[:, 3], mask[:, 3])
                    loss += wh_loss_func(mask[:, 0] * out[:, 4], mask[:, 4])
                    pbar.set_postfix({'loss (batch)': loss.item()})
                    pbar.update(img.shape[0])
                    val_loss += loss
                val_loss /= (len(val_data) / batch_size)
                logging.info(f"val_loss = {val_loss}")
                ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(),
                           os.path.join(save_path, ckpt_name))
                logging.info(f"Save model {epoch}!")


if __name__ == "__main__":
    train()
