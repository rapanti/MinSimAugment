import itertools as it

import torch
import torch.nn.functional as nnf


def select_crops_identity(images, student, teacher, criterion, fp16, num_crops, epoch):
    return images, torch.zeros(1), torch.zeros(1)


@torch.no_grad()
def select_crops_cross(images, student, teacher, criterion, fp16, num_crops, epoch):
    b, c, h, w = images[0].shape
    device = student.device

    with torch.cuda.amp.autocast(fp16 is not None):
        teacher_output = teacher(images[:4])
        student_output = student(images)
    student_output, teacher_output = criterion.prepare_outputs(student_output, teacher_output, epoch)
    student_output, teacher_output = student_output.chunk(len(images)), teacher_output.chunk(4)

    out = [torch.zeros_like(images[0]) for _ in range(2)] + [torch.zeros_like(images[-1]) for _ in range(2)]

    score = torch.full([b], 0, device=device)
    selected = torch.zeros((4, b), dtype=torch.uint8)

    global_combinations = [(0, 1), (0, 3), (2, 1), (2, 3)]
    # instead of it.combinations(range(4), 2)
    # we use global_combinations to avoid "illegal" global views, because pair has same augmentations
    for idg, idl in it.product(global_combinations, it.combinations(range(4, 8), 2)):
        _teacher_out = [teacher_output[idg[0]], teacher_output[idg[1]]]
        _student_out = [student_output[idg[0]], student_output[idg[1]], student_output[idl[0]], student_output[idl[1]]]
        with torch.cuda.amp.autocast(fp16 is not None):
            sim = criterion.select_forward(_student_out, _teacher_out)  # sample-loss
            score, indices = torch.stack((score, sim)).max(dim=0)
            indices = indices.type(torch.bool)
        selected[0][indices] = idg[0]
        selected[1][indices] = idg[1]
        selected[2][indices] = idl[0]
        selected[3][indices] = idl[1]
        for n, ids in enumerate(idg + idl):
            out[n] = torch.where(indices[:, None, None, None], images[ids], out[n])

    return out, selected, score


@torch.no_grad()
def select_crops_avgpool(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.avgpool_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_conv1_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.conv1_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_first_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.first_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_second_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.second_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_anchor(images, model, fp16):
    with torch.cuda.amp.autocast(fp16 is not None):
        embeds = [model.module.simple_forward(img) for img in images]
    p1, z1 = embeds[0]
    scores = [nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1) for p2, z2 in embeds[1:]]
    stacked = torch.stack(scores)
    values, indic = stacked.min(dim=0)
    a = nnf.one_hot(indic, len(scores)).T
    out = torch.zeros_like(images[0])
    for n, img in enumerate(images[1:]):
        out += a[n].view(-1, 1, 1, 1) * img
    return images[0], out


names = {
    "anchor": select_crops_anchor,
    "cross": select_crops_cross,
    "conv1layer": select_crops_conv1_layer,
    "firstlayer": select_crops_first_layer,
    "secondlayer": select_crops_second_layer,
    "avgpool": select_crops_avgpool,
    "identity": select_crops_identity,
}