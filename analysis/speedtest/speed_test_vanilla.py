import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import models
import torch.utils.benchmark as benchmark

import builder

if __name__ == "__main__":
    bs = 64

    model = builder.SimSiam(
        models.resnet50,
        dim=2048,
        pred_dim=512,
        proj_layer=3,
    ).cuda()

    criterion = nn.CosineSimilarity(dim=1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-6)

    fp16_scaler = GradScaler()

    x = torch.randn(bs, 3, 224, 224).cuda()
    y = torch.randn(bs, 3, 224, 224).cuda()

    def test_fn(x1, x2):
        for _ in range(100):
            
            with autocast():
                p1, p2, z1, z2 = model(x1, x2)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            optimizer.zero_grad()
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

    t = benchmark.Timer(
        stmt="test_fn(x, y)",
        globals={"test_fn": test_fn, "x": x, "y": y},
    )

    print(t.timeit(1))