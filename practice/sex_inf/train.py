import logging
import torch
import os
import torch.nn as nn
from dataset import StatureDataset
from torch.utils.data import DataLoader, random_split
from net import SexInf,SexInf_3D
from tqdm import tqdm


def train():
    csv_path = "../test_waist.csv"
    save_path ="./out_waistline"
    epochs = 100
    val_percent = 0.2
    batch_size = 2048
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SexInf_3D()
    critirion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    net.to(device=device)
    dataset = StatureDataset(csv_path)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_data, val_data = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    for epoch in range(epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for data, label in train_loader:
                data = data.to(device=device)
                label = label.to(device=device)
                pred = net(data)
                loss = critirion(pred, label)
                pbar.set_postfix({'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(data.shape[0])
        scheduler.step()
        val_loss = 0
        net.eval()
        with torch.no_grad():
            with tqdm(total=n_val, desc="val round", unit='batch', leave=False) as pbar:
                for data, label in val_loader:
                    data = data.to(device=device)
                    label = label.to(device=device)
                    pred = net(data)
                    loss = critirion(pred, label)
                    val_loss += loss.item()
                    pbar.set_postfix({'loss (batch)': loss.item()})
                    pbar.update(data.shape[0])
        val_loss /= (n_val/batch_size)
        logging.info(f"loss={val_loss}")
        ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
        torch.save(net.state_dict(),
                   os.path.join(save_path, ckpt_name))
        logging.info(f"Save model {epoch}!")


if __name__ == "__main__":
    train()
