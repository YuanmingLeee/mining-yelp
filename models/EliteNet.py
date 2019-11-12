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
        self.drop_1 = nn.Dropout(C.drop1)
        self.fc2 = nn.Linear(C.fc2, C.fc3, bias=True)
        self.bn_2 = nn.BatchNorm1d(C.fc3)
        self.drop_2 = nn.Dropout(C.drop2)
        self.fc3 = nn.Linear(C.fc3, C.fc4, bias=True)
        self.out = nn.Linear(C.fc4, C.output, bias=True)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.bn_1(self.fc1(x)))
        x = self.drop_1(x)
        x = torch.tanh(self.bn_2(self.fc2(x)))
        x = self.drop_2(x)
        x = torch.tanh(self.fc3(x))
        out = self.out(x)

        return out
