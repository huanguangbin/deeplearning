import os
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from net.model import NestedUNet
from losses import *
from dataset import BasicDataset
from config import UNetConfig
from glob import glob
from metrics import dice_coef, iou_score
from collections import OrderedDict
from AverageMeter import AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
cfg = UNetConfig()


def validate(cfg, val_loader, model, criterion, n_val, device):
    losses_value = AverageMeter()
    ious_value = AverageMeter()
    dices_value = AverageMeter()

    model.eval()

    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation', unit='img(s)', leave=False) as pbar:
            for i, batch in enumerate(val_loader):
                loss = 0
                batch_imgs = batch['image']
                batch_masks = batch['mask']
                assert batch_imgs.shape[1] == cfg.n_channels, "channels configuration not correct"
                batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
                batch_masks = batch_masks.to(device=device, dtype=mask_type)

                output = model(batch_imgs)
                if cfg.deepsupervision:
                    for inference in output:
                        loss += criterion(inference, batch_masks)
                    loss /= len(output)
                    iou = iou_score(output[-1], batch_masks, cfg.out_threshold)
                    dice = dice_coef(output[-1], batch_masks, cfg.out_threshold)
                else:
                    loss = criterion(output, batch_masks)
                    iou = iou_score(output, batch_masks, cfg.out_threshold)
                    dice = dice_coef(output, batch_masks, cfg.out_threshold)

                pbar.set_postfix({'loss (batch)': loss.item()})
                pbar.update(batch_imgs.size(0))

                losses_value.update(loss.item(), batch_imgs.size(0))
                ious_value.update(iou, batch_imgs.size(0))
                dices_value.update(dice, batch_imgs.size(0))

    log = OrderedDict([
        ('loss', losses_value.avg),
        ('iou', ious_value.avg),
        ('dice', dices_value.avg)
    ])

    return log


def train():
    cfg = UNetConfig()
    writer = SummaryWriter(comment="OCTSeg")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"use device {device}")
    model = eval(cfg.model)(cfg)
    model = model.to(device=device)
    model = torch.nn.DataParallel(model)
    if cfg.pretrain_model is not None:
        model.module.load_state_dict(
            OrderedDict({k.replace('module.', ''): v for k, v in torch.load(cfg.pretrain_model).items()}))
        # model.load_state_dict(torch.load(args.pretrain_model))
    criterion = nn.BCEWithLogitsLoss().to(device=device) if cfg.loss == 'BCEWithLogitsLoss' else eval(cfg.loss)().to(
        device=device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    data_paths = glob(os.path.join(cfg.images_dir, "*"))
    mask_paths = glob(os.path.join(cfg.masks_dir, "*"))

    # train_data_paths, val_data_paths, train_mask_paths, val_mask_paths = \
    #     train_test_split(data_paths, mask_paths, test_size=0.2, random_state=41)
    logging.info(f"train_num:{str(len(data_paths))}")
    logging.info(f"creating model {cfg.model}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.lr) if cfg.optimizer == 'Adam' else optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)

    dataset = BasicDataset(data_paths, mask_paths, aug=cfg.aug)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(45))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    best_iou = 0
    trigger = cfg.early_stop
    batch_step = 0
    for epoch in range(cfg.epochs):
        # if trigger <= args.early_stop * (1-0.6) and :
        #     criterion = losses.__dict__["LovaszHingeLoss"]().to(device=device)
        #     logging.info("Use LovaszHingeLoss")
        #     trigger = args.early_stop
        losses_value = AverageMeter()
        ious_value = AverageMeter()
        dices_value = AverageMeter()
        model.train()
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit="img(s)") as pbar:
            for i, batch in enumerate(train_loader):
                loss = 0
                batch_imgs = batch['image']
                batch_masks = batch['mask']
                assert batch_imgs.shape[1] == cfg.n_channels, "channels configuration not correct"
                batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
                batch_masks = batch_masks.to(device=device, dtype=mask_type)

                output = model(batch_imgs)
                if cfg.deepsupervision:
                    for inference in output:
                        loss += criterion(inference, batch_masks)
                    loss /= len(output)
                    iou = iou_score(output[-1], batch_masks, cfg.out_threshold)
                    dice = dice_coef(output[-1], batch_masks, cfg.out_threshold)
                else:
                    loss = criterion(output, batch_masks)
                    iou = iou_score(output, batch_masks, cfg.out_threshold)
                    dice = dice_coef(output, batch_masks, cfg.out_threshold)

                pbar.set_postfix({'loss (batch)': loss.item(), 'iou (batch)': iou, 'dice (batch)': dice})
                writer.add_scalar(tag="train/batch_loss", scalar_value=loss.item(), global_step=batch_step)

                losses_value.update(loss.item(), batch_imgs.size(0))
                ious_value.update(iou, batch_imgs.size(0))
                dices_value.update(dice, batch_imgs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(batch_imgs.size(0))
                batch_step += 1

        train_log = OrderedDict([
            ('loss', losses_value.avg),
            ('iou', ious_value.avg),
            ('dice', dices_value.avg)
        ])

        writer.add_scalar(tag="train/epoch_loss", scalar_value=train_log['loss'], global_step=epoch)
        val_log = validate(cfg, val_loader, model, criterion, len(val_dataset), device)
        writer.add_scalar(tag="validation/Dice", scalar_value=val_log['dice'], global_step=epoch)
        writer.add_scalar(tag="validation/Iou", scalar_value=val_log['iou'], global_step=epoch)
        logging.info('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
                     % (
                         train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'],
                         val_log['dice']))

        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(),
                       os.path.join(cfg.checkpoints_dir, "model_" + str(epoch) + ".pth"))
        else:
            torch.save(model.state_dict(),
                       os.path.join(cfg.checkpoints_dir, "model_" + str(epoch) + ".pth"))

        if val_log['iou'] > best_iou:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(cfg.checkpoints_dir, "model_bst.pth"))
            else:
                torch.save(model.module.state_dict(), os.path.join(cfg.checkpoints_dir, "model_bst.pth"))
            best_iou = val_log['iou']
            logging.info("Saved best model")
            trigger = cfg.early_stop
        else:
            trigger -= 1
            logging.info(f"trigger={trigger}")

        if cfg.early_stop is not None:
            if trigger == 0:
                logging.info("early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
