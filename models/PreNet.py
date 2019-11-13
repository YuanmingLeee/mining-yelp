import torch
from torch import nn


def _freeze(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class PreNet(nn.Module):
    def __init__(self, backbone: nn.Module, profiling: nn.Module, pretrained=True):
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.profiling = nn.Sequential(*list(profiling.children())[:-1])
        self.fc1 = nn.Linear(512, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 2, bias=True)

        # freeze if pretrained flag is set
        if pretrained:
            _freeze(backbone)
            _freeze(profiling)

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.profiling(x2)
        x = torch.cat((x1, x2), dim=1)
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.out(x)
        return x
