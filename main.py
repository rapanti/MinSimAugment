from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf
from torch.backends import cudnn

import utils


parser = ArgumentParser("ImageNet ResNet50 Benchmarks")
parser.add_argument("--data_path", type=Path, default="/datasets/imagenet/train")
parser.add_argument("--output_dir", type=Path, default=".")
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")


def main(cfg):
    pass


def pretrain():
    pass


if __name__ == '__main__':
    cfg = OmegaConf.load('pretrain.yaml')
    gpu, rank, world_size = utils.init_distributed_mode(
        cfg.distributed.backend, cfg.distributed.dist_url)
    utils.fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    main(cfg)
