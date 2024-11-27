import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

__all__ = ["transposed_conv_net"]

class Residual(nn.Module):  
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        return self.layers(x)
    
class TransConvNet(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):

        assert len(mid_channels) == 4
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[1]),
        )
        self.transposed_conv1d = nn.Sequential(
            nn.Unflatten(-1, (-1, 1)),
            nn.ConvTranspose1d(
                in_channels=mid_channels[1], out_channels=mid_channels[2], kernel_size=30, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=mid_channels[2], out_channels=mid_channels[3], kernel_size=30, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=mid_channels[3], out_channels=2, kernel_size=30, stride=1, padding=0),
        )
        L = 1
        for i in range(3):
            L = calculate_Lout(L, 30)

        self.predictor = nn.Sequential(
            nn.Linear(L, mid_channels[4]),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(mid_channels[4], mid_channels[4]),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(mid_channels[4], self.out_channels)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.transposed_conv1d(x)
        x = self.predictor(x)
        return x

# class transposed_conv_net(nn.Module):
    
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.fc1 = nn.Linear(in_channels, 100)
#         self.relu1 = nn.ReLU()
#         self.res_1 = Residual(100, 200)
#         self.transposed_conv1d = nn.Sequential(
#             nn.Unflatten(-1, (-1, 1)),
#             nn.ConvTranspose1d(
#                 in_channels=100, out_channels=30, kernel_size=40, stride=1, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose1d(
#                 in_channels=30, out_channels=30, kernel_size=40, stride=1, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose1d(
#                 in_channels=30, out_channels=2, kernel_size=40, stride=1, padding=0),
#             nn.ReLU(),
#         )
#         L = 1
#         for i in range(3):
#             L = calculate_Lout(L, 40)
#         self.predictor = nn.Sequential(
#             nn.Linear(L, 500),
#             nn.BatchNorm1d(2),
#             nn.ReLU(),
#             nn.Linear(500, out_channels)
#         )

#     def forward(self, x):
#         x = self.fc1(x)
#         y = self.relu1(x)
#         x = self.res_1(y) + x
#         x = self.transposed_conv1d(x)
#         x = self.predictor(x)
#         return x


if __name__ == "__main__":
    x = torch.randn(20, 3)
    net = transposed_conv_net(3, 501)
    y = net(x)
    print(y.shape)