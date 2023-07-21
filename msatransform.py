import torch.nn as nn
import torchvision.transforms.functional as tf
from torchvision.transforms import Compose
from torchvision.transforms import RandomResizedCrop

import numpy as np


def calc_iou(p1, p2):
    def get_dims(x):
        return x[0], x[0] + x[2], x[1], x[1] + x[3]

    def calc_overlap(x1, x2):
        l1, r1, t1, b1 = get_dims(x1)
        l2, r2, t2, b2 = get_dims(x2)
        w = max(0, min(r1, r2) - max(l1, l2))
        h = max(0, min(b1, b2) - max(t1, t2))
        return w * h

    def calc_area(x):
        w, h = x[-2:]
        return w * h

    A = calc_area(p1)
    B = calc_area(p2)
    o = calc_overlap(p1, p2)
    return o / (A + B - o)


class MSATransform(nn.Module):
    def __init__(self,
                 rrc: RandomResizedCrop,
                 total_epochs: int,
                 start_val: float = 0.5,
                 end_val: float = None,
                 transforms: Compose = None,
                 ):
        super().__init__()
        self.epoch = 0
        self.total_epochs = total_epochs

        if end_val is None:
            end_val = start_val
        self.schedule = np.linspace(start_val, end_val, total_epochs)
        self.start_val = start_val
        self.end_val = end_val

        self.rrc = rrc
        self.transforms = transforms

    def forward(self, img):
        img1 = self.rrc(img)
        img2 = self.rrc(img)
        for _ in range(111):
            i1, j1, h1, w1 = self.rrc.get_params(img, self.rrc.scale, self.rrc.ratio)
            i2, j2, h2, w2 = self.rrc.get_params(img, self.rrc.scale, self.rrc.ratio)

            val = calc_iou((i1, j1, h1, w1), (i2, j2, h2, w2))
            if val > self.schedule[self.epoch]:
                continue
            img1 = tf.resized_crop(img, i1, j1, h1, w1, self.rrc.size, self.rrc.interpolation, antialias=self.rrc.antialias)
            img2 = tf.resized_crop(img, i2, j2, h2, w2, self.rrc.size, self.rrc.interpolation, antialias=self.rrc.antialias)
            break

        return self.transforms(img1), self.transforms(img2)

    def set_epoch(self, epoch):
        self.epoch = epoch
