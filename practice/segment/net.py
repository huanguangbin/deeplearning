import torch.nn as nn


class NetSem(nn.Module):
    def __init__(self):
        super(NetSem, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.final = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        out = self.conv4(conv3_out)
        out = self.final(out)
        return out
