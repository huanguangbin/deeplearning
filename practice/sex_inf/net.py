import torch.nn as nn


class SexInf(nn.Module):

    def __init__(self):
        super(SexInf, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class SexInf_3D(nn.Module):

    def __init__(self):
        super(SexInf_3D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(3, 30),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
