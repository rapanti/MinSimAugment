import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import models
import torch.utils.benchmark as benchmark

import builder

@torch.no_grad()
def select_crops_cross(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.single_forward(torch.cat(images, dim=0))
    p1s, z1s = model_out[0].chunk(len(images)), model_out[1].chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)
    selected = torch.zeros((2, b), dtype=torch.uint8)

    for n in range(len(images)):
        p1, z1 = p1s[n], z1s[n]
        for m in range(n + 1, len(images)):
            p2, z2 = p1s[m], z1s[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = (nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1)) * 0.5
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)
            selected[0][indices] = n
            selected[1][indices] = m
            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2, selected, score



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

    images = [torch.randn(bs, 3, 224, 224).cuda() for _ in range(4)]

    def test_fn(images):
        for _ in range(100):

            x, y, _, _ = select_crops_cross(images, model, fp16_scaler)

            with autocast():
                p1, p2, z1, z2 = model(x, y)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            optimizer.zero_grad()
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

    
    t = benchmark.Timer(
        stmt="test_fn(images)",
        globals={"test_fn": test_fn, "images": images},
    )

    print(t.timeit(1))