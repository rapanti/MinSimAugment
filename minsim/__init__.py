import torch


@torch.no_grad()
def identity(images, model, fp16):
    return images, torch.zeros(1), torch.zeros(1)


@torch.no_grad()
def cross(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        p1s, z1s = model(torch.cat(images, dim=0))
    p1s, z1s = p1s.chunk(len(images)), z1s.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)
    selected = torch.zeros((2, b), dtype=torch.uint8)

    for n in range(len(images)):
        x1 = p1s[n], z1s[n]
        for m in range(n + 1, len(images)):
            x2 = p1s[m], z1s[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = model.loss_fn(x1, x2, reduction='none')
                score, indices = torch.stack((score, sim)).max(dim=0)
                indices = indices.type(torch.bool)
            selected[0][indices] = n
            selected[1][indices] = m
            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return (out1, out2), selected, score
