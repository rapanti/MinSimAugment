import itertools as it

import torch


_NUM_GLOBAL_CROPS_LOADER = -1
_NUM_LOCAL_CROPS_LOADER = -1
_NUM_LOCAL_CROPS = -1
_LIMIT_COMBINATIONS = 0
_COMBINATIONS = []


def setup(num_global_crops_loader, num_local_crops_loader, num_local_crops, limit_combinations):
    global _NUM_GLOBAL_CROPS_LOADER, _NUM_LOCAL_CROPS_LOADER, _NUM_LOCAL_CROPS, _LIMIT_COMBINATIONS, _COMBINATIONS

    _NUM_GLOBAL_CROPS_LOADER = num_global_crops_loader
    _NUM_LOCAL_CROPS_LOADER = num_local_crops_loader
    _NUM_LOCAL_CROPS = num_local_crops
    _LIMIT_COMBINATIONS = limit_combinations

    g_comb = it.product(range(0, _NUM_GLOBAL_CROPS_LOADER, 2), range(1, _NUM_GLOBAL_CROPS_LOADER, 2))
    l_comb = it.combinations(
        range(_NUM_GLOBAL_CROPS_LOADER, _NUM_GLOBAL_CROPS_LOADER + _NUM_LOCAL_CROPS_LOADER), _NUM_LOCAL_CROPS,
    )
    all_comb = it.product(g_comb, l_comb)
    _COMBINATIONS = [gl + ll for gl, ll in all_comb]


@torch.no_grad()
def hardviews(images, student, teacher, criterion, fp16, epoch):
    bs = images[0].size(0)
    device = student.device

    score = torch.zeros(bs, device=device)
    selected = torch.zeros((2 + _NUM_LOCAL_CROPS, bs), dtype=torch.uint8, device=device)
    out = [torch.empty_like(images[0]) for _ in range(2)]
    out += [torch.empty_like(images[-1]) for _ in range(_NUM_LOCAL_CROPS)]

    with torch.cuda.amp.autocast(fp16 is not None):
        student_output = student(images)
        teacher_output = teacher(images[:_NUM_GLOBAL_CROPS_LOADER])
        student_output, teacher_output = criterion.prepare_outputs(
            student_output, teacher_output, epoch
        )
        student_output, teacher_output = (
            student_output.chunk(len(images)),
            teacher_output.chunk(_NUM_GLOBAL_CROPS_LOADER),
        )

    for idx in _combinations(_LIMIT_COMBINATIONS):
        _teacher_out = [teacher_output[x] for x in idx[:2]]
        _student_out = [student_output[x] for x in idx]
        with torch.cuda.amp.autocast(fp16 is not None):
            sim = criterion.select_forward(_student_out, _teacher_out)  # sample-loss
            score, indices = torch.stack((score, sim)).max(dim=0)
            indices = indices.type(torch.bool)

        for n, ids in enumerate(idx):
            selected[n][indices] = ids

    for n in range(2):
        for m in range(_NUM_GLOBAL_CROPS_LOADER):
            out[n] = torch.where(
                (selected[n] == m)[:, None, None, None], images[m], out[n]
            )
    for n in range(2, len(out)):
        for m in range(_NUM_LOCAL_CROPS_LOADER, len(images)):
            out[n] = torch.where(
                (selected[n] == m)[:, None, None, None], images[m], out[n]
            )

    # check that all images are selected correctly
    # for n, ids in enumerate(selected):
    #     for m, idx in enumerate(ids):
    #         assert torch.equal(out[n][m], images[idx][m])

    return out, selected, score


def _combinations(limit):
    if limit:
        for idx in torch.randint(len(_COMBINATIONS), (limit,)):
            yield _COMBINATIONS[idx]
    else:
        for comb in _COMBINATIONS:
            yield comb
