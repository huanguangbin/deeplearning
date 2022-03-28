# -*- coding: utf-8 -*-
import os
import torch
import unet3d
import losses
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from glob import glob
from tqdm import tqdm
from unet3d import UNet3D
from vnet1 import VNet
#from vnet3d import VNet
from tensorboardX import SummaryWriter
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import Dataset
from metrics import dice_coef, iou_score
from utils import str2bool, count_params

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
arch_names = list(unet3d.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', '-a', metavar='ARCH', default="VNet",#'UNet3D'
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')

    parser.add_argument('--vol_data_path', default="/home/test/hgb/data/Intestinal/3D_imgs_npy_714", type=str)
    parser.add_argument('--vol_mask_path', default="/home/test/hgb/data/Intestinal/3D_masks_npy_714", type=str)
    parser.add_argument('--val_vol_data_path', default="/home/test/hgb/data/Intestinal/val_imgs_714", type=str)
    parser.add_argument('--val_vol_mask_path', default="/home/test/hgb/data/Intestinal/val_msks_714", type=str)
    # parser.add_argument('--vol_data_path', default="/home/test/hgb/data/Intestinal/ori/train/imgs", type=str)
    # parser.add_argument('--vol_mask_path', default="/home/test/hgb/data/Intestinal/ori/train/msks", type=str)
    # parser.add_argument('--val_vol_data_path', default="/home/test/hgb/data/Intestinal/ori/train/valset/imgs", type=str)
    # parser.add_argument('--val_vol_mask_path', default="/home/test/hgb/data/Intestinal/ori/train/valset/msks", type=str)

    parser.add_argument('--pretrain_model', default=None, type=str,
                        help="pretrain model path")
    parser.add_argument('--model_save_path', default="./output/VNet_attention", type=str)
    parser.add_argument('--model_name', default="vnet_att", type=str)
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=30, type=int,
                        metavar='N', help='early stopping')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='nesterov')
    parser.add_argument('--aug', default=True, type=str2bool)
    parser.add_argument('--threshold', default=0.5, type=float, help="value must between 0 and 1")
    args = parser.parse_args()

    return args


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(args, val_loader, model, criterion, n_val):
    losses_value = AverageMeter()
    ious_value = AverageMeter()
    dices_value = AverageMeter()

    model.eval()

    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation', unit='volData', leave=False) as pbar:
            for i, (vol_data, vol_mask) in enumerate(val_loader):
                vol_data = vol_data.cuda()
                vol_mask = vol_mask.cuda()

                output = model(vol_data)
                loss = criterion(output, vol_mask)
                iou = iou_score(output, vol_mask, args.threshold)
                dice = dice_coef(output, vol_mask, args.threshold)
                pbar.set_postfix({'loss (batch)': loss.item()})
                pbar.update(vol_data.size(0))

                losses_value.update(loss.item(), vol_data.size(0))
                ious_value.update(iou, vol_data.size(0))
                dices_value.update(dice, vol_data.size(0))

    log = OrderedDict([
        ('loss', losses_value.avg),
        ('iou', ious_value.avg),
        ('dice', dices_value.avg)
    ])

    return log


def train():
    args = parse_args()
    writer = SummaryWriter(comment="OCTSeg")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"use device {device}")
    model = VNet()
    model = model.to(device=device)
    model = torch.nn.DataParallel(model)
    if args.pretrain_model is not None:
        # model._initialize_weights()
        model.module.load_state_dict(
            OrderedDict({k.replace('module.', ''): v for k, v in torch.load(args.pretrain_model).items()}))
        # model.load_state_dict(torch.load(args.pretrain_model))
    criterion = nn.BCEWithLogitsLoss().to(device=device) if args.loss == 'BCEWithLogitsLoss' else losses.__dict__[
        args.loss]().to(device=device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    train_data_paths = glob(os.path.join(args.vol_data_path, "*"))
    train_mask_paths = glob(os.path.join(args.vol_mask_path, "*"))
    val_data_paths = glob(os.path.join(args.val_vol_data_path, "*"))
    val_mask_paths = glob(os.path.join(args.val_vol_mask_path, "*"))

    # train_data_paths, val_data_paths, train_mask_paths, val_mask_paths = \
    #     train_test_split(data_paths, mask_paths, test_size=0.2, random_state=41)
    logging.info(f"train_num:{str(len(train_data_paths))}")
    logging.info(f"val_num:{str(len(val_data_paths))}")
    logging.info(f"creating model {args.arch}")
    logging.info(f"model params:{count_params(model)}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr) if args.optimizer == 'Adam' else optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_data_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_data_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    best_iou = 0
    trigger = args.early_stop
    batch_step = 0
    for epoch in range(args.epochs):
        # if trigger <= args.early_stop * (1-0.6) and :
        #     criterion = losses.__dict__["LovaszHingeLoss"]().to(device=device)
        #     logging.info("Use LovaszHingeLoss")
        #     trigger = args.early_stop
        losses_value = AverageMeter()
        ious_value = AverageMeter()
        dices_value = AverageMeter()
        model.train()
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{args.epochs}', unit="volData") as pbar:
            for i, (vol_data, vol_mask) in enumerate(train_loader):
                vol_data = vol_data.to(device=device)
                vol_mask = vol_mask.to(device=device)
                output = model(vol_data)

                loss = criterion(output, vol_mask)
                iou = iou_score(output, vol_mask, args.threshold)
                dice = dice_coef(output, vol_mask, args.threshold)
                pbar.set_postfix({'loss (batch)': loss.item()})
                writer.add_scalar(tag="train/batch_loss", scalar_value=loss.item(), global_step=batch_step)

                losses_value.update(loss.item(), vol_data.size(0))
                ious_value.update(iou, vol_data.size(0))
                dices_value.update(dice, vol_data.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(vol_data.size(0))
                batch_step += 1

        train_log = OrderedDict([
            ('loss', losses_value.avg),
            ('iou', ious_value.avg),
            ('dice', dices_value.avg)
        ])

        writer.add_scalar(tag="train/epoch_loss", scalar_value=train_log['loss'], global_step=epoch)
        val_log = validate(args, val_loader, model, criterion, len(val_dataset))
        writer.add_scalar(tag="validation/Dice", scalar_value=val_log['dice'], global_step=epoch)
        writer.add_scalar(tag="validation/Iou", scalar_value=val_log['iou'], global_step=epoch)
        logging.info('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
                     % (
                         train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'],
                         val_log['dice']))

        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(),
                       os.path.join(args.model_save_path, args.model_name + "_" + str(epoch) + ".pth"))
        else:
            torch.save(model.state_dict(),
                       os.path.join(args.model_save_path, args.model_name + "_" + str(epoch) + ".pth"))

        if val_log['iou'] > best_iou:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(args.model_save_path, args.model_name + ".pth"))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, args.model_name + ".pth"))
            best_iou = val_log['iou']
            logging.info("Saved best model")
            trigger = args.early_stop
        else:
            trigger -= 1
            logging.info(f"trigger={trigger}")

        if args.early_stop is not None:
            if trigger == 0:
                logging.info("early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
