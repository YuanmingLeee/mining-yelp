import torch
from torch import nn

from configs import cfg


class EliteNet(nn.Module):
    def __init__(self):
        super().__init__()
        # get config
        C = cfg.user_elite

        # structure
        self.input = nn.Linear(C.input, C.fc1, bias=True)
        self.fc1 = nn.Linear(C.fc1, C.fc2, bias=True)
        self.bn_1 = nn.BatchNorm1d(C.fc2)
        self.fc2 = nn.Linear(C.fc2, C.fc3, bias=True)
        self.bn_2 = nn.BatchNorm1d(C.fc3)
        self.out = nn.Linear(C.fc3, C.output, bias=True)

    def forward(self, x):
        hidden = torch.tanh(self.input(x))
        fc1 = torch.tanh(self.bn_1(self.fc1(hidden)))
        fc2 = torch.tanh(self.bn_2(self.fc2(fc1)))
        out = self.out(fc2)

        return out
