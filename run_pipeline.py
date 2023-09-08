from omegaconf import OmegaConf
import eval_linear
import pretrain
import utils

if __name__ == "__main__":
    # load pretrain config
    cfg = OmegaConf.load('pretrain.yaml')

    # init distributed
    utils.distributed.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.seed)

    print('*************STARTING PRETRAINING*************')
    pretrain.main(cfg)

    print('*************STARTING LINEAR EVAL EVALUATION: ImageNet*************')
    eval_linear_cfg = OmegaConf.load("eval_linear.yaml")
    # copy dist parameters
    eval_linear_cfg.gpu = cfg.gpu
    eval_linear_cfg.rank = cfg.rank
    eval_linear_cfg.world_size = cfg.world_size
    eval_linear_cfg.dist_url = cfg.dist_url
    eval_linear.main(eval_linear_cfg)

    print('*************STARTING FINETUNING: CIFAR10*************')
    eval_linear_cfg.dataset = "CIFAR10"
    eval_linear_cfg.batch_size = 512
    eval_linear_cfg.finetune = True
    eval_linear_cfg.lr = 7.5e-6
    eval_linear_cfg.weight_decay = 0.05
    eval_linear_cfg.optimizer = "adamw"
    eval_linear_cfg.epochs = 300
    eval_linear_cfg.data_path = "../datasets"
    eval_linear.main(eval_linear_cfg)

    print('*************STARTING FINETUNING: CIFAR100*************')
    eval_linear_cfg.dataset = "CIFAR100"
    eval_linear_cfg.batch_size = 512
    eval_linear_cfg.finetune = True
    eval_linear_cfg.lr = 5e-6
    eval_linear_cfg.weight_decay = 0.05
    eval_linear_cfg.optimizer = "adamw"
    eval_linear_cfg.epochs = 300
    eval_linear_cfg.data_path = "../datasets"
    eval_linear.main(eval_linear_cfg)

    print('*************STARTING LINEAR EVAL EVALUATION: Flowers102*************')
    eval_linear_cfg.dataset = "Flowers102"
    eval_linear_cfg.batch_size = 512
    eval_linear_cfg.finetune = True
    eval_linear_cfg.lr = 5e-4
    eval_linear_cfg.weight_decay = 0.05
    eval_linear_cfg.optimizer = "adamw"
    eval_linear_cfg.epochs = 300
    eval_linear_cfg.data_path = "../datasets"
    eval_linear.main(eval_linear_cfg)

    print('*************STARTING LINEAR EVAL EVALUATION: Places365*************')
    eval_linear_cfg.dataset = "Places365"
    eval_linear_cfg.batch_size = 512
    eval_linear_cfg.finetune = True
    eval_linear_cfg.lr = 5e-5
    eval_linear_cfg.weight_decay = 0.05
    eval_linear_cfg.optimizer = "adamw"
    eval_linear_cfg.epochs = 300
    eval_linear_cfg.data_path = "/work/dlclarge1/ferreira-simsiam/minsim_experiments/datasets/places365"
    eval_linear.main(eval_linear_cfg)

    print('*************STARTING LINEAR EVAL EVALUATION: iNaturalist (using train_mini)*************')
    eval_linear_cfg.dataset = "inat21"
    eval_linear_cfg.batch_size = 512
    eval_linear_cfg.lr = 5e-5
    eval_linear_cfg.weight_decay = 0.005
    eval_linear_cfg.epochs = 100
    eval_linear_cfg.optimizer = "adamw"
    eval_linear_cfg.data_path = "/work/dlclarge1/ferreira-simsiam/minsim_experiments/datasets"
    eval_linear.main(eval_linear_cfg)
