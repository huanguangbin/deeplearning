import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StatureDataset(Dataset):
    def __init__(self, csv):
        frame = pd.read_csv(csv)
        df = pd.DataFrame(frame)
        self.data = np.array(df)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.Tensor(self.data[item][1:]), torch.Tensor([self.data[item][0]])


class PlotDataset(Dataset):
    def __init__(self, h, w, waistline=None):
        self.weight_data = w.reshape(-1)
        self.height_data = h.reshape(-1)
        if waistline is not None:
            self.waistline_data = waistline.reshape(-1)
        else:
            self.waistline_data = None

    def __len__(self):
        return self.weight_data.shape[0]

    def __getitem__(self, item):
        data = np.append(self.height_data[item], self.weight_data[item])
        if self.waistline_data is not None:
            data = np.append(data, self.waistline_data[item])
        return torch.Tensor(data)
