import torch
import pandas as pd
import numpy as np
from dataset import PlotDataset
from torch.utils.data import DataLoader
from net import SexInf
import matplotlib.pyplot as plt


def plot_2D():
    delta = 0.5
    csv_path = "../test1.csv"
    model_path = "./out/epoch_58.pth"
    batch_size = 2048
    frame = pd.read_csv(csv_path)
    df = pd.DataFrame(frame)
    arr = np.array(df)
    h_min, h_max = np.min(arr[:, 1]) - 1, np.max(arr[:, 1]) + 1
    w_min, w_max = np.min(arr[:, 2]) - 1, np.max(arr[:, 2]) + 1
    h, w = np.meshgrid(np.arange(h_min, h_max, delta), np.arange(w_min, w_max, delta))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SexInf()
    net.to(device=device)
    net.load_state_dict(torch.load(model_path))
    dec = 0
    dataset = PlotDataset(h, w)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for idx, data in enumerate(dataloader):
        data = data.to(device=device)
        pred = net(data).detach().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8).reshape(-1)
        if idx == 0:
            dec = pred
        else:
            dec = np.append(dec, pred)
    dec = dec.reshape(h.shape[0], h.shape[1])
    plt.contourf(h, w, dec)
    plt.scatter(arr[:, 1], arr[:, 2], c=arr[:, 0])
    plt.show()


def plot_3D():
    delta = 0.5
    csv_path = "../test_waist.csv"
    model_path = "./out_waistline/epoch_99.pth"
    batch_size = 2048
    frame = pd.read_csv(csv_path)
    df = pd.DataFrame(frame)
    arr = np.array(df)
    h_min, h_max = np.min(arr[:, 1]) - 1, np.max(arr[:, 1]) + 1
    w_min, w_max = np.min(arr[:, 2]) - 1, np.max(arr[:, 2]) + 1
    h, w = np.meshgrid(np.arange(h_min, h_max, delta), np.arange(w_min, w_max, delta))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SexInf()
    net.to(device=device)
    net.load_state_dict(torch.load(model_path))
    dec = 0
    dataset = PlotDataset(h, w)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for idx, data in enumerate(dataloader):
        data = data.to(device=device)
        pred = net(data).detach().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8).reshape(-1)
        if idx == 0:
            dec = pred
        else:
            dec = np.append(dec, pred)
    dec = dec.reshape(h.shape[0], h.shape[1])
    plt.contourf(h, w, dec, aalpha=0.1)
    plt.scatter(arr[:, 1], arr[:, 2], c=arr[:, 0])
    plt.show()


if __name__ == "__main__":
    plot_2D()
