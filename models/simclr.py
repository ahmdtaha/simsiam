import torch
import torch.nn as nn
from models.cifar_resnet import resnet18 as cifar_resnet18




class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512,out_dim=512):
        super().__init__()
        # hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def get_backbone(cfg, castrate=True):
    # backbone = eval(f"{backbone}(zero_init_residual=True)")
    backbone = cfg.backbone
    if cfg.set in ['CIFAR10','CIFAR100']:
        backbone = 'cifar_resnet18'
    backbone = eval(f"{backbone}(zero_init_residual=True)")
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

class SimCLR(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        self.projector = projection_MLP(self.backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return (z1,z2)




















