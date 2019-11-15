import torch
from torch import nn

from configs import parse_config, BASE_DIR
from models.EliteNet import EliteNet
from models.TextLSTM import TextLSTM


def _freeze(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class PreNet(nn.Module):
    def __init__(self, config: str, pretrained: bool = True):
        super().__init__()
        self.cfg = parse_config(config)
        self.backbone: nn.Module = TextLSTM(BASE_DIR / self.cfg.BACKBONE.NET).cuda()
        if getattr(self.cfg.BACKBONE, 'WEIGHT', None):
            self.backbone.load_state_dict(torch.load(BASE_DIR / self.cfg.BACKBONE.WEIGHT))
        self.backbone.fc = nn.Identity()
        self.backbone.sfm = nn.Identity()

        profiling = EliteNet(BASE_DIR / self.cfg.PROFILING.NET).cuda()
        if getattr(self.cfg.BACKBONE, 'WEIGHT', None):
            self.backbone.load_state_dict(torch.load(BASE_DIR / self.cfg.PROFILING.WEIGHT))
        self.profiling = nn.Sequential(*list(profiling.children())[:-1])

        self.fc1 = nn.Linear(512, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 2, bias=True)

        # freeze if pretrained flag is set
        if pretrained:
            _freeze(self.backbone)
            _freeze(self.profiling)

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.profiling(x2)
        x = torch.cat((x1, x2), dim=1)
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.out(x)
        return x
