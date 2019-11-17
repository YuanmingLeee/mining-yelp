import torch
from torch import nn
from torch.utils import data as tdata

from parser import parse_config


class EliteNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # get config
        self.cfg = parse_config(config)

        # structure
        self.input = nn.Linear(self.cfg.INPUT, self.cfg.FC1, bias=True)
        self.fc1 = nn.Linear(self.cfg.FC1, self.cfg.FC2, bias=True)
        self.bn_1 = nn.BatchNorm1d(self.cfg.FC2)
        self.drop_1 = nn.Dropout(self.cfg.DROP1)
        self.fc2 = nn.Linear(self.cfg.FC2, self.cfg.FC3, bias=True)
        self.bn_2 = nn.BatchNorm1d(self.cfg.FC3)
        self.drop_2 = nn.Dropout(self.cfg.DROP2)
        self.fc3 = nn.Linear(self.cfg.FC3, self.cfg.FC4, bias=True)
        self.out = nn.Linear(self.cfg.FC4, self.cfg.OUTPUT, bias=True)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.bn_1(self.fc1(x)))
        x = self.drop_1(x)
        x = torch.tanh(self.bn_2(self.fc2(x)))
        x = self.drop_2(x)
        x = torch.tanh(self.fc3(x))
        out = self.out(x)

        return out

    def batch_predict(self, data_loader: tdata.DataLoader):
        pred = []
        for samples in data_loader:
            features = samples['features'].cuda()
            labels = samples['label'].cuda()

            # forward
            output = self.__call__(features)
            pred += torch.argmax(output, dim=1).tolist()
        return pred
