import argparse
import json
import math
import os
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

import data
import distributed as dist
import builder
import utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(cfg):
    dist.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    print(f"git:\n  {utils.get_sha()}\n")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(cfg)).items())))

    if cfg.dataset == "CIFAR10":
        global models
        import resnet_cifar as models

    model = builder.SimSiam(
        models.__dict__[cfg.arch],
        cfg.dim, cfg.pred_dim
    ).cuda()

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])

    init_lr = cfg.lr * cfg.batch_size / 256

    criterion = nn.CosineSimilarity(dim=1).cuda()

    if cfg.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    fp16 = torch.cuda.amp.GradScaler() if cfg.fp16 else None

    # ============ preparing data ... ============
    if cfg.dataset == "CIFAR10":
        mean, std = data.CIFAR10_DEFAULT_MEAN, data.CIFAR10_DEFAULT_STD
    else:
        mean, std = data.IMAGENET_DEFAULT_MEAN, data.IMAGENET_DEFAULT_STD

    transform = data.make_pretrain_transform(
        crop_size=cfg.crop_size,
        crop_scale=cfg.crop_scale,
        blur_prob=cfg.blur_prob,
        hflip_prob=cfg.hflip_prob,
        mean=mean,
        std=std,
    )
    two_crops_transform = data.TwoCropsTransform(transform)
    dataset, _ = data.make_dataset(cfg.data_path, cfg.dataset, True, two_crops_transform)

    sampler = DistributedSampler(dataset)
    cfg.batch_size_per_gpu = cfg.batch_size // dist.get_world_size()
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "total_time": 0}
    utils.restart_from_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16=fp16,
    )
    start_epoch = to_restore["epoch"]
    total_time = to_restore["total_time"]

    log_dir = os.path.join(cfg.output_dir, "tensorboard")
    board = SummaryWriter(log_dir) if dist.is_main_process() else None

    for epoch in range(start_epoch, cfg.epochs):
        loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        start = time.time()
        train_stats = train(loader, model, criterion, optimizer, epoch, cfg, fp16, board)
        total_time += int(time.time() - start)

        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": cfg,
            "fp16": fp16.state_dict() if fp16 is not None else None,
            "total_time": total_time,
        }
        utils.save_on_master(save_dict, os.path.join(cfg.output_dir, 'checkpoint.pth'))
        if cfg.saveckp_freq and epoch and epoch % cfg.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(cfg.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if dist.is_main_process():
            with (Path(cfg.output_dir) / "pretrain.log").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(loader, model, criterion, optimizer, epoch, cfg, fp16, board):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(loader, cfg.print_freq, header)):
        it = len(loader) * epoch + it  # global training iteration

        x1 = images[0].cuda(non_blocking=True)
        x2 = images[1].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16 is not None):
            p1, p2, z1, z2 = model(x1=x1, x2=x2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        optimizer.zero_grad()
        if fp16 is None:
            loss.backward()
            optimizer.step()
        else:
            fp16.scale(loss).backward()
            fp16.step(optimizer)
            fp16.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if dist.is_main_process() and it % cfg.logger_freq == 0:
            board.add_scalar("training loss", loss.item(), it)
            board.add_scalar("training lr", optimizer.param_groups[0]["lr"], it)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def get_args_parser():
    p = argparse.ArgumentParser("SimSiam", description='Pytorch Pretraining on ImageNet', add_help=False)
    p.add_argument('--dataset', default="ImageNet", type=str)
    p.add_argument('--data_path', type=str, help='path to training data.')
    p.add_argument('-a', '--arch', default="resnet50", type=str)
    p.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    p.add_argument('-b', '--batch_size', default=512, type=int,
                   help='total batch-size (default: 512)')
    p.add_argument('--lr', default=0.05, type=float, help='initial (base) learning rate')
    p.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
    p.add_argument('--wd', '--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)',
                   dest="weight_decay")
    p.add_argument('--fp16', default=True, type=utils.bool_flag,
                   help="Whether or not to use half precision for training.")

    # simsiam specific configs:
    p.add_argument('--dim', default=2048, type=int,
                   help='feature dimension (default: 2048)')
    p.add_argument('--pred_dim', default=512, type=int,
                   help='hidden dimension of the predictor (default: 512)')
    p.add_argument('--fix_pred_lr', default=True, type=utils.bool_flag,
                   help='Fix learning rate for the predictor')

    # data augmentation configs:
    p.add_argument("--crop_size", default=224, type=int, help="Size of the crops.")
    p.add_argument("--crop_scale", default=(0.2, 1.0), type=float, nargs='+', help="Size of the crops.")
    p.add_argument("--blur_prob", default=0.5, type=float, help="Blur probability.")
    p.add_argument("--hflip_prob", default=0.5, type=float, help="Horizontal Flip.")

    # Misc
    p.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    p.add_argument('--saveckp_freq', default=0, type=int, help='Save checkpoint every x epochs.')
    p.add_argument('--seed', default=0, type=int, help='Random seed.')
    p.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    p.add_argument("--dist_backend", default="nccl", type=str, help="Distributed backend.")
    p.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    p.add_argument("--print_freq", default=10, type=int, help="Print progress every x iterations.")
    p.add_argument("--logger_freq", default=50, type=int, help="Log progress every x iterations to tensorboard.")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
