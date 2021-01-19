#
# Video Action Recognition with Pytorch
#
# Paper citation
#
# Action Recognition in Video Sequences using
# Deep Bi-Directional LSTM With CNN Features
# 2017, AMIN ULLAH et al.
# Digital Object Identifier 10.1109/ACCESS.2017.2778011 @ IEEEAccess
#
# See also main.py
#

import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

def linear(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

class LSTM_with_CNN(nn.Module):
    def __init__(self):
        super(LSTM_with_CNN, self).__init__()
        self.conv1 = conv(3, 96, (1, 11, 11), (1, 4, 4), (0, 4, 4))
        self.conv2 = conv(96, 256, (1, 5, 5), 1, (0, 2, 2))
        self.conv3 = conv(256, 384, (1, 3, 3), 1, (0, 1, 1))
        self.conv4 = conv(384, 384, (1, 3, 3), 1, (0, 1, 1))
        self.conv5 = conv(384, 256, (1, 3, 3), 1, (0, 1, 1))
        self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.fc6 = linear(256*4*4, 4096)
        self.fc7 = linear(4096, 4096)
        self.fc8 = linear(4096, 1000)
        self.lstm = nn.LSTM(1000, 51, 2, batch_first=True, bidirectional=True)
        self.linear = linear(102, 51)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape = torch.Size([batch_size, 3, frames, 112, 112])

        # CNN
        x = self.conv1(x)
        # x.shape = torch.Size([batch_size, 96, frames, 28, 28])
        x = self.maxpool(x)
        # x.shape = torch.Size([batch_size, 96, frames, 14, 14])
        x = self.conv2(x)
        # x.shape = torch.Size([batch_size, 256, frames, 14, 14])
        x = self.maxpool(x)
        # x.shape = torch.Size([batch_size, 256, frames, 7, 7])
        x = self.conv3(x)
        # x.shape = torch.Size([batch_size, 384, frames, 7, 7])
        x = self.conv4(x)
        # x.shape = torch.Size([batch_size, 384, frames, 7, 7])
        x = self.conv5(x)
        # x.shape = torch.Size([batch_size, 256, frames, 7, 7])
        x = self.maxpool(x)
        # x.shape = torch.Size([batch_size, 256, frames, 4, 4])

        # FC Layers
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # x.shape = torch.Size([batch_size, frames, 256, 4, 4])
        x = x.view(-1, 256*4*4)
        # x.shape = torch.Size([batch_size * frames, 4096])
        x = self.fc6(x)
        # x.shape = torch.Size([batch_size * frames, 4096])
        x = self.fc7(x)
        # x.shape = torch.Size([batch_size * frames, 4096])
        x = self.fc8(x)
        # x.shape = torch.Size([batch_size * frames, 1000])

        # RNN(BD-LSTM)
        x = x.view(batch_size, -1, 1000)
        # x.shape = torch.Size([batch_size, frames, 1000])
        x, hidden = self.lstm(x)
        # x.shape = torch.Size([batch_size, frames, 51 * 2])

        # FC Layer and Softmax
        frames = x.shape[1]
        x = x[:, frames - 1, :]
        # x.shape = torch.Size([batch_size, 51 * 2])
        x = self.linear(x)
        # x.shape = torch.Size([batch_size, 51])
        x = self.softmax(x)
        # x.shape = torch.Size([batch_size, 51])

        return x
