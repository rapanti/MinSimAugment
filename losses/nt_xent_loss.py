import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor

import distributed as dist
from losses.utils import gather


class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature: float, gather_distributed: bool = True):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(self, z0: Tensor, z1: Tensor, reduction="mean") -> Tensor:
        z0 = nn.functional.normalize(z0, p=2, dim=1)
        z1 = nn.functional.normalize(z1, p=2, dim=1)

        if self.gather_distributed and dist.get_world_size() > 1:
            z0 = gather(z0)
            z1 = gather(z1)

        z = torch.cat((z0, z1), dim=0)
        sim = nnf.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature

        self_mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        pos_mask = self_mask.roll(shifts=sim.size(0) // 2, dims=0)

        sim.masked_fill_(self_mask, -torch.inf)
        positive_samples = sim[pos_mask].reshape(sim.size(0), 1)
        negative_samples = sim[~pos_mask].reshape(sim.size(0), -1)

        labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return nnf.cross_entropy(logits, labels, reduction=reduction)
