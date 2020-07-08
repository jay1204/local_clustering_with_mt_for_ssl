import torch
import torch.nn as nn
from model.model_helper import *


class ShallowNet(nn.Module):
    def __init__(self, num_classes):
        super(ShallowNet, self).__init__()

        conv_layers = self.conv_block_1(3, 128, drop_out=0.0) + \
                      self.conv_block_1(128, 256, drop_out=0.5) + \
                      self.conv_block_2(256, 128, drop_out=0.5)

        self.conv_blocks = nn.Sequential(*conv_layers)

        self.global_pool = nn.AvgPool2d(6, stride=1, padding=0)
        self.fc = WN_Linear_Mean_Only_BN(128, num_classes, train_scale=True, init_stdv=0.1)

    def conv_block_1(self, in_channels, out_channels, drop_out):
        conv_block = [
            nn.Dropout(drop_out),

            WN_Conv2d_Mean_Only_BN(in_channels, out_channels, kernel_size=3, stride=1, padding=1, train_scale=True),
            nn.LeakyReLU(0.1, inplace=True),

            WN_Conv2d_Mean_Only_BN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, train_scale=True),
            nn.LeakyReLU(0.1, inplace=True),

            WN_Conv2d_Mean_Only_BN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, train_scale=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(2, stride=2, padding=0),
        ]

        return conv_block

    def conv_block_2(self, in_channels, out_channels, drop_out):
        conv_block = [
            nn.Dropout(drop_out),

            WN_Conv2d_Mean_Only_BN(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=0, train_scale=True),
            nn.LeakyReLU(0.1, inplace=True),

            WN_Conv2d_Mean_Only_BN(out_channels * 4, out_channels * 2, kernel_size=1, stride=1, padding=0, train_scale=True),
            nn.LeakyReLU(0.1, inplace=True),

            WN_Conv2d_Mean_Only_BN(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, train_scale=True),
            nn.LeakyReLU(0.1, inplace=True),
        ]

        return conv_block

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x, self.fc(x)
