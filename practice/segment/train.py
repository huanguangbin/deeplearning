import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader, random_split
from dataset import SemDataset
from net import NetSem
from metrics import *
import copy

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--datapath', default=r"./data", help='data path')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--use_cuda', default=True, help='using CUDA for training')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True


def train():
    os.makedirs('./output', exist_ok=True)

    dataset = SemDataset(os.path.join(args.datapath, "images"), os.path.join(args.datapath, "masks"),
                         transform=transforms.ToTensor())
    val_percent = 0.2
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_data, val_data = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    model = NetSem()

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_iou = []
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            n_class = out.size(1)
            out = out.permute((0, 2, 3, 1))
            out = out.contiguous().view(-1, out.size(-1))
            label = batch_y.permute((0, 2, 3, 1))
            label = label.contiguous().view(-1, label.size(-1)).squeeze(-1).to(dtype=torch.long)
            loss = loss_func(out, label)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            batch_iou = []
            for cls in range(1, n_class + 1):
                cls_pred = torch.zeros_like(pred)
                cls_pred[pred == cls] = 1
                cls_label = torch.zeros_like(label)
                cls_label[label == cls] = 1
                batch_iou.append(iou_score(cls_pred, cls_label))
            if len(train_iou) == 0:
                train_iou = copy.deepcopy(batch_iou)
            else:
                train_iou = [i + j for i, j in zip(train_iou, batch_iou)]
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f,Iou1: %.3f,Iou2: %.3f,Iou3: %.3f'
                  % (epoch + 1, args.epochs, batch, math.ceil(len(train_data) / args.batch_size),
                     loss.item(), batch_iou[0], batch_iou[1], batch_iou[2]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: {}, Iou1: {},Iou2: {},Iou3: {}'.format(
            train_loss / (math.ceil(len(train_data) / args.batch_size)),
            train_iou[0] * args.batch_size / (len(train_data)), train_iou[1] * args.batch_size / (len(train_data)),
            train_iou[2] * args.batch_size / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_iou = []
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            n_class = out.size(1)
            out = out.permute((0, 2, 3, 1))
            out = out.contiguous().view(-1, out.size(-1))
            label = batch_y.permute((0, 2, 3, 1))
            label = label.contiguous().view(-1, label.size(-1)).squeeze(-1).to(dtype=torch.long)
            loss = loss_func(out, label)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            batch_iou = []
            for cls in range(1, n_class + 1):
                cls_pred = torch.zeros_like(pred)
                cls_pred[pred == cls] = 1
                cls_label = torch.zeros_like(label)
                cls_label[label == cls] = 1
                batch_iou.append(iou_score(cls_pred, cls_label))
            if len(eval_iou) == 0:
                eval_iou = copy.deepcopy(batch_iou)
            else:
                eval_iou = [i + j for i, j in zip(eval_iou, batch_iou)]
        print('Val Loss: {}, Iou1: {},Iou2: {},Iou3: {}'.format(
            eval_loss / (math.ceil(len(train_data) / args.batch_size)),
            eval_iou[0] * args.batch_size / (len(val_data)), eval_iou[1] * args.batch_size / (len(val_data)),
            eval_iou[2] * args.batch_size / (len(val_data))))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')
            # to_onnx(model, 3, 28, 28, 'params.onnx')


if __name__ == '__main__':
    train()
