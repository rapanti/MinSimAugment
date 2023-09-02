from omegaconf import OmegaConf

import eval_linear, eval_knn
import pretrain
import utils

if __name__ == "__main__":
    # load pretrain config
    cfg = OmegaConf.load('pretrain.yaml')

    # init distributed
    utils.distributed.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)

    print('STARTING PRETRAINING')
    pretrain.main(cfg)

    print('STARTING LINEAR EVAL EVALUATION')
    eval_linear_cfg = OmegaConf.load("eval_linear.yaml")
    # copy dist parameters
    eval_linear_cfg.gpu = cfg.gpu
    eval_linear_cfg.rank = cfg.rank
    eval_linear_cfg.world_size = cfg.world_size
    eval_linear_cfg.dist_url = cfg.dist_url
    eval_linear.main(eval_linear_cfg)

    print('STARTING LINEAR EVAL EVALUATION iNaturalist')
    eval_linear_cfg = OmegaConf.load("eval_linear.yaml")
    # copy dist parameters
    eval_linear_cfg.dataset = "inat18"
    eval_linear_cfg.data_path = "../datasets"
    eval_linear.main(eval_linear_cfg)

    print('STARTING LINEAR EVAL EVALUATION Flowers102')
    eval_linear_cfg = OmegaConf.load("eval_linear.yaml")
    # copy dist parameters
    eval_linear_cfg.dataset = "Flowers102"

    eval_linear_cfg.batch_size = eval_linear_cfg.batch_size//2
    eval_linear_cfg.data_path = "../datasets"
    eval_linear.main(eval_linear_cfg)

    print('STARTING LINEAR EVAL EVALUATION StanfordCars')
    eval_linear_cfg = OmegaConf.load("eval_linear.yaml")
    # copy dist parameters
    eval_linear_cfg.dataset = "StanfordCars"
    eval_linear_cfg.batch_size = eval_linear_cfg.batch_size // 2
    eval_linear_cfg.data_path = "../datasets"
    eval_linear.main(eval_linear_cfg)