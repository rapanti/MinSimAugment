from omegaconf import OmegaConf

import eval_linear
import pretrain


if __name__ == "__main__":
    cfg = OmegaConf.load('pretrain.yaml')
    print('STARTING PRETRAINING')
    pretrain.main(cfg)

    eval_cfg = OmegaConf.load('eval_linear.yaml')
    # copy dist parameters
    eval_cfg.gpu = cfg.gpu
    eval_cfg.rank = cfg.rank
    eval_cfg.world_size = cfg.world_size
    # we assume the random master_port chosen for pretraining is still available for eval
    # even if machine has changed
    eval_cfg.dist_url = cfg.dist_url
    print('STARTING EVALUATION')
    eval_linear.main(eval_cfg)
