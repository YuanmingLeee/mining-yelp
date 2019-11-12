from torch import nn


class PreNet(nn.Module):
    def __init__(self, backbone: nn.Module, profiling: nn.Module):
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.profiling = nn.Sequential(*list(profiling.children())[:-1])
