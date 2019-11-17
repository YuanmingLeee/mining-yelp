import torch
from torch import nn
from torch.utils import data as tdata

from configs import BASE_DIR
from models.EliteNet import EliteNet
from models.TextLSTM import TextLSTM
from parser import parse_config


def _freeze(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class MultimodalClassifier(nn.Module):
    def __init__(self, config: str, pretrained: bool = True):
        super().__init__()
        self.cfg = parse_config(config)
        self.backbone = TextLSTM(BASE_DIR / self.cfg.BACKBONE.NET).cuda()
        if getattr(self.cfg.BACKBONE, 'WEIGHT', None):
            self.backbone.load_state_dict(torch.load(BASE_DIR / self.cfg.BACKBONE.WEIGHT))
        self.backbone.fc = nn.Identity()
        self.backbone.sfm = nn.Identity()

        profiling = EliteNet(BASE_DIR / self.cfg.PROFILING.NET).cuda()
        if getattr(self.cfg.BACKBONE, 'WEIGHT', None):
            profiling.load_state_dict(torch.load(BASE_DIR / self.cfg.PROFILING.WEIGHT))
        self.profiling = nn.Sequential(*list(profiling.children())[:-1])

        bottle_neck = self.backbone.cfg.HIDDEN_SIZE + profiling.cfg.FC4

        self.fc1 = nn.Linear(bottle_neck, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512, bias=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, 2, bias=True)

        # freeze if pretrained flag is set
        if pretrained:
            _freeze(self.backbone)
            _freeze(self.profiling)

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.profiling(x2)
        x = torch.cat((x1, x2), dim=1)
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x

    def batch_predict(self, data_loader: tdata.DataLoader):
        pred = []
        gt = []
        for samples in data_loader:
            elite = samples['elite'].cuda()
            text = samples['text'].cuda()
            labels = samples['label'].cuda()
            gt += labels.tolist()

            # forward
            output = self.__call__(text, elite)
            pred += torch.argmax(output, dim=1).tolist()
        return pred, gt
