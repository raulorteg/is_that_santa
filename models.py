import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

class SantaFinder(nn.Module):
    def __init__(self, input_shape:torch.Tensor):
        super(SantaFinder, self).__init__()
        self.batch_size = input_shape[0]
        self.img_channels = input_shape[1]
        self.img_height = input_shape[2]
        self.img_width = input_shape[3]

        # expected to work with [3,128,128]
        self.cnn = nn.Sequential(
            # [3,128,128] -> [32,63,63] 
            nn.Conv2d(in_channels=self.img_channels, out_channels=32, kernel_size=8, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # [32,63,63] -> [64,31,31] 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # [64,31,31] -> [128,15,15] 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # [128,15,15] -> [128,8,8]
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # [128,8,8] -> [128,4,4]
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # [128,4,4] -> 2048
        self.flatten = nn.Flatten(start_dim=1)

        # 2048 -> 1
        self.linear = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        xhat = self.cnn(x)
        xhat = self.flatten(xhat)
        out = self.linear(xhat)
        return {"preds":out}