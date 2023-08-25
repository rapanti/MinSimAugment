import torch
import torch.nn as nn
import torchvision
from utils import off_diagonal
import torch.distributed


class BarlowTwins(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, cfg.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        # (16, 8192) -> 8 * (2, 8192)
        #

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)  # (batch_size, dim_bn).T @ (batch_size, dim_bn) -> dim_bn x dim_bn
        # (8192, 16) @ (16, 8192) -> (8192, 8192)
        #

        # sum the cross-correlation matrix between all gpus
        c.div_(self.cfg.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.cfg.lambd * off_diag
        return loss

    @torch.no_grad()
    def single_forward(self, x):
        z = self.backbone(x)
        p = self.projector(z)
        return p

    @torch.no_grad()
    def conv1_layer(self, x):
        return self.encoder.conv1_layer(x)

    @torch.no_grad()
    def first_layer(self, x):
        return self.encoder.first_layer(x)

    @torch.no_grad()
    def second_layer(self, x):
        return self.encoder.second_layer(x)

    @torch.no_grad()
    def avgpool_layer(self, x):
        return self.encoder.avgpool_layer(x)
