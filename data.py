import os
import sys
from typing import Sequence, Tuple

import torchvision.transforms
import torchvision.transforms.functional as F
from torchvision.transforms import CenterCrop, ColorJitter, Compose, GaussianBlur, InterpolationMode,  \
    Normalize, RandomApply, RandomGrayscale, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, VOCDetection, INaturalist, Places365, \
    FakeData, Flowers102, StanfordCars, INaturalist, Food101


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)

FLOWERS102_DEFAULT_MEAN = (0.4353, 0.3773, 0.2872)
FLOWERS102_DEFAULT_STD = (0.2966, 0.2455, 0.2698)


class MultiCropsTransform:
    """Take multiple random crops of one image as the query and key."""

    def __init__(self, base_transform, num_crops):
        self.base_transform = base_transform
        self.num_crops = num_crops

    def __call__(self, x):
        images = []
        params = []
        for _ in range(self.num_crops):
            # img, p = self.base_transform(x)
            img = torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.))(x)
            # TrivialAugment does not return params
            img = self.base_transform(img)
            img = F.to_tensor(img)
            images.append(img)
            params.append([])
        return images, params


def make_normalize_transform(
    mean: Sequence[float], std: Sequence[float],
) -> Normalize:
    return Normalize(mean=mean, std=std)


def make_pretrain_transform(
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.2, 1.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        blur_prob: float = 0.5,
        hflip_prob: float = 0.5,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        RandomResizedCrop(crop_size, scale=crop_scale, interpolation=interpolation),
        RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        RandomGrayscale(p=0.2),
    ]
    if blur_prob > 0.0:
        transforms_list.append(RandomApply([GaussianBlur(9, (0.1, 2.0))], p=blur_prob))
    if hflip_prob > 0.0:
        transforms_list.append(RandomHorizontalFlip(p=hflip_prob))
    transforms_list.extend(
        [
            ToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return Compose(transforms_list)


def make_classification_train_transform(
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.08, 1.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        hflip_prob: float = 0.5,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        RandomResizedCrop(crop_size, scale=crop_scale, interpolation=interpolation),
        RandomHorizontalFlip(p=hflip_prob),
        ToTensor(),
        make_normalize_transform(mean=mean, std=std)
    ]
    return Compose(transforms_list)


def make_classification_val_transform(
        resize_size: int = 256,
        crop_size: int = 224,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        Resize(resize_size, interpolation=interpolation),
        CenterCrop(crop_size),
        ToTensor(),
        make_normalize_transform(mean=mean, std=std)
    ]
    return Compose(transforms_list)



def make_dataset(
        root: str,
        dataset: str,
        train: bool,
        transform):
    if dataset == 'CIFAR10':
        return CIFAR10(root, download=False, train=train, transform=transform), 10
    elif dataset == 'CIFAR100':
        return CIFAR100(root, download=False, train=train, transform=transform), 100
    elif dataset == 'Food101':
        split = "train" if train else "test"
        return Food101(root, download=False, split=split, transform=transform), 101
    elif dataset == "Flowers102":
        split = "train" if train else "test"
        return Flowers102(root, download=False, split=split, transform=transform), 102
    elif dataset == "StanfordCars":
        split = "train" if train else "test"
        return StanfordCars(root, download=True, split=split, transform=transform), 196
    elif dataset == "inat21":
        version = "2021_train_mini" if train else "2021_valid"
        return INaturalist(root, download=False, version=version, transform=transform), 10000
    elif dataset == "Places365":
        split = "train-standard" if train else "val"
        return Places365(root, download=False, split=split, transform=transform), 365
    elif dataset == 'ImageNet':
        root = os.path.join(root, 'train' if train else 'val')
        dataset = ImageFolder(root, transform=transform)
        return dataset, 1000
    elif dataset == 'Test32':
        dataset = FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform)
        return dataset, 10
    elif dataset == 'Test224':
        dataset = FakeData(size=1000, image_size=(3, 224, 224), num_classes=1000, transform=transform)
        return dataset, 1000
    print(f"Does not support dataset: {dataset}")
    sys.exit(1)