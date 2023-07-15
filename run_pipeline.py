from omegaconf import OmegaConf
from utils import find_free_port

import eval_linear
import pretrain


if __name__ == "__main__":
    # Define master port (for preventing 'Address already in use error' when submitting more than 1 jobs on 1 node)
    master_port = find_free_port()
    cfg = OmegaConf.load('pretrain.yaml')
    cfg.dist_url = "tcp://localhost:" + str(master_port)
    print('STARTING PRETRAINING')
    pretrain.main(cfg)

    eval_cfg = OmegaConf.load('eval_linear.yaml')
    # copy dist parameters
    eval_cfg.gpu = cfg.gpu
    eval_cfg.rank = cfg.rank
    eval_cfg.world_size = cfg.world_size
    eval_cfg.dist_url = cfg.dist_url
    print('STARTING EVALUATION')
    eval_linear.main(eval_cfg)
