import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self, length, n_classes):
        super(SimpleCNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.n_classes = n_classes
        self.length = length

        self.conv0 = nn.Conv2d(3, 16, (5, 5), padding=2)
        self.conv1 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.bn0 = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

        self.lin0   = nn.Linear(10368, 128)
        self.lin1   = nn.Linear(128, length*n_classes)

    def forward(self, x):
        o = F.relu(self.conv0(x))
        o = F.max_pool2d(o, (3,3))
        o = self.bn0(o)

        o = F.relu(self.conv1(o))
        o = F.max_pool2d(o, (2,2))
        o = self.bn1(o)

        o = F.relu(self.conv2(o))
        o = F.max_pool2d(o, (2,2))
        o = self.bn2(o)

        # flatten non-batch dimensions
        o = o.view(o.size(0), -1)

        o = F.relu(self.lin0(o))
        o = self.lin1(o) 

        o = o.view(o.size(0), self.n_classes, self.length)

        return o

