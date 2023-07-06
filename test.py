from omegaconf import OmegaConf

import eval_linear
import pretrain


if __name__ == "__main__":
    cfg = OmegaConf.load('testing/pretrain.yaml')
    pretrain.main(cfg)
