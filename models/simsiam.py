import torch
import torch.nn as nn
from models.backbones import *
from models.cifar_resnet import resnet18 as cifar_resnet18

class projection_MLP(nn.Module):
    def __init__(self,cfg, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        if cfg.set in ['CIFAR10','CIFAR100']:
            cfg.logger.info('CIFAR Identity layer')
            self.layer2 = nn.Identity()
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def get_backbone(cfg, castrate=True):
    backbone = cfg.backbone
    if cfg.set in ['CIFAR10','CIFAR100']:
        backbone = 'cifar_resnet18'
    backbone = eval(f"{backbone}(zero_init_residual=True)")
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

class SimSiam(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = get_backbone(cfg)
        self.projector = projection_MLP(cfg,self.backbone.output_dim,hidden_dim=cfg.emb_dim,out_dim=cfg.emb_dim)

        # self.encoder = nn.Sequential(  # f encoder
        #     self.backbone,
        #     self.projector
        # )
        self.predictor = prediction_MLP(in_dim=cfg.emb_dim,out_dim=cfg.emb_dim)

    def forward(self, x1, x2):

        bb = self.backbone
        f = self.projector
        h =  self.predictor

        bb1, bb2 = bb(x1), bb(x2)
        z1, z2 = f(bb1), f(bb2)
        p1, p2 = h(z1), h(z2)
        # L = D(p1, z2) / 2 + D(p2, z1) / 2
        return ((bb1,z1,p1),(bb2,z2,p2))

