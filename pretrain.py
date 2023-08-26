import argparse
import json
import math
import os
import time
import datetime
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from methods.simsiam import SimSiam
from transforms.simsiam_transform import SimSiamTransform

import data
from utils import dist as dist
import minsim
import utils


def main(cfg):
    dist.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    print(f"git:\n  {utils.get_sha()}\n")
    print(OmegaConf.to_yaml(cfg))

    backbone = models.__dict__[cfg.arch](zero_init_residual=True)

    model = SimSiam(backbone).cuda()

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])

    init_lr = cfg.lr * cfg.batch_size / 256
    optimizer = model.module.configure_optimizers(cfg.fix_pred_lr, init_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    fp16 = torch.cuda.amp.GradScaler() if cfg.fp16 else None

    # ============ preparing data ... ============
    transform = SimSiamTransform(return_params=True)
    dataset, _ = data.make_dataset(cfg.data_path, cfg.dataset, True, transform)

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

    # ============ preparing MinSim ... ============
    ms_fn = minsim.names[cfg.select_fn]

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
        train_stats, metrics = train(loader, model, optimizer, epoch, cfg, fp16, board, ms_fn)
        total_time += int(time.time() - start)

        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
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
        if dist.is_main_process():
            with (Path(cfg.output_dir) / "metrics.json").open("a") as f:
                f.write(json.dumps(metrics) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(loader, model, optimizer, epoch, cfg, fp16, board, ms_fn):
    model.train()
    metrics = defaultdict(list)
    metrics["epoch"] = epoch
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.epochs)
    for it, batch in enumerate(metric_logger.log_every(loader, cfg.print_freq, header)):
        it = len(loader) * epoch + it  # global training iteration

        images = batch[0][:cfg.num_crops]
        params = batch[0][cfg.num_crops:]

        images = [img.cuda(non_blocking=True) for img in images]

        images, selected, sample_loss = ms_fn(images, model, fp16)

        with torch.cuda.amp.autocast(fp16 is not None):
            loss = model.module.training_step(images, it)

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

        metrics["loss"].append(loss.item())
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        if cfg.use_adv_metric and it % cfg.adv_metric_freq == 0:
            params = [p.tolist() for p in params]
            metrics["selected"].append(selected.tolist())
            metrics["params"].append(params)
            metrics["sample-loss"].append(sample_loss.tolist())

        if dist.is_main_process() and it % cfg.logger_freq == 0:
            board.add_scalar("training loss", loss.item(), it)
            board.add_scalar("training lr", optimizer.param_groups[0]["lr"], it)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metrics


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
    p.add_argument('--dataset', type=str, default="ImageNet",
                   help='Specify dataset (default: ImageNet)')
    p.add_argument('--data_path', type=str,
                   help='(root) path to dataset')
    p.add_argument('-a', '--arch', type=str,
                   help="Name of architecture to train (default: resnet50)")
    p.add_argument('--epochs', type=int,
                   help='number of total epochs to run (default: 100)')
    p.add_argument('-b', '--batch_size', type=int,
                   help='total batch-size (default: 512)')
    p.add_argument('--lr', type=float,
                   help='initial (base) learning rate (default: 0.05)')
    p.add_argument('--momentum', type=float,
                   help='momentum of SGD solver (default: 0.9)')
    p.add_argument('--wd', '--weight_decay', dest="weight_decay", type=float,
                   help='weight decay (default: 1e-4)')

    # simsiam specific parameters:
    p.add_argument('--dim', type=int,
                   help='feature dimension (default: 2048)')
    p.add_argument('--pred_dim', type=int,
                   help='hidden dimension of the predictor (default: 512)')
    p.add_argument('--fix_pred_lr', type=utils.bool_flag,
                   help='Fix learning rate for the predictor (default: True)')

    # data augmentation parameters:
    p.add_argument("--crop_size", type=int,
                   help="Size of crops (default: 224)")
    p.add_argument("--crop_scale", type=float, nargs='+',
                   help="Scale range of the crops, relative to original image (default: 0.2 1.)")
    p.add_argument("--blur_prob", type=float,
                   help="Blur probability (default: 0.5)")
    p.add_argument("--hflip_prob", type=float,
                   help="Horizontal-Flip probability (default: 0.5)")

    # MinSim parameters:
    p.add_argument("--num_crops", default=2, type=int, help="Number of crops")
    p.add_argument("--select_fn", default="identity", type=str)

    # Misc
    p.add_argument('--fp16', default=True, type=utils.bool_flag,
                   help="Whether or not to use half precision for training. (default: True)")
    p.add_argument('--output_dir', default=".", type=str,
                   help='Path to save logs and checkpoints.')
    p.add_argument('--saveckp_freq', default=0, type=int,
                   help='Save checkpoint every x epochs.')
    p.add_argument('--seed', default=0, type=int,
                   help='Random seed.')
    p.add_argument('--num_workers', default=8, type=int,
                   help='Number of data loading workers per GPU.')
    p.add_argument("--dist_backend", default="nccl", type=str,
                   help="distributed backend (default: nccl)")
    p.add_argument("--dist_url", default="env://", type=str,
                   help="url used to set up distributed training")
    p.add_argument("--print_freq", default=10, type=int,
                   help="Print progress every x iterations (default: 10)")
    p.add_argument("--logger_freq", default=50, type=int,
                   help="Log progress every x iterations to tensorboard (default: 50)")
    p.add_argument("--use_adv_metric", default=False, type=utils.bool_flag,
                   help="Log advanced metrics: transforms params, crop selection, sample-loss, ... (default: False)")
    p.add_argument("--adv_metric_freq", default=100, type=int,
                   help="Log advanced metrics every x iterations (default: 100)")

    return p


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
