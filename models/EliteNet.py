import torch.nn.functional as F
from torch import nn


class EliteNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(15, 512, bias=True)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.bn_1 = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.bn_2 = nn.BatchNorm2d(256)
        self.out = nn.Linear(256, 2, bias=True)

    def forward(self, x):
        hidden = F.tanh(self.input(x))
        fc1 = F.tanh(self.bn_1(self.fc1(hidden)))
        fc2 = F.tanh(self.bn_2(self.fc2(fc1)))
        out = self.out(fc2)

        return out
