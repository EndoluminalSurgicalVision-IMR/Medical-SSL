import torchvision.transforms.functional as tf
import torch
from torchvision import transforms
import numpy as np
from scipy import ndimage
import random
import os
import csv
from datasets_2D.paths import Path
from datasets_2D.Classification.base_classification import ClassificationBase
import argparse
from torch.utils.data import DataLoader
import utils.tools as utils
from PIL import Image
from torchvision.transforms import transforms, ToTensor
from packaging import version



class ClassificationEyePACSSet(ClassificationBase):
    """
   dataset for segmentation.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/valid/test
        """
        super(ClassificationEyePACSSet, self).__init__(config, base_dir, flag)

        self.all_images = []
        self.all_labels = []
        if self.flag == 'train':
            self.root_dir = os.path.join(self._base_dir,  'train_1024')
        else:
            self.root_dir = os.path.join(self._base_dir,  'test_1024')

        with open(os.path.join(self._base_dir, flag + '.csv')) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            for row in reader:
                self.all_images.append(self.root_dir+'/'+row[1]+'.jpg')
                self.all_labels.append(int(row[2]))
            f.close()

        assert len(self.all_images) != 0, "the images can`t be zero!"
        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

        if hasattr(self.config, 'data_aug') and self.flag == 'train':
            aug_cfg = data_augmentation_args
            self.transform = data_transforms(self.config, aug_cfg)
        else:
            self.transform = simple_transform(self.config.input_size)


class data_augmentation_args:
    horizontal_flip_prob=0.5
    vertical_flip_prob=0.5
    color_distortion_prob= 1.0
    brightness= 0.2
    contrast=0.2
    saturation= 0
    hue= 0
    random_crop_prob = 1.0 
    # randomly crop and resize tinput_size
    random_crop_scale = [0.87, 1.15]
    # range of size of the origin size cropped
    random_crop_ratio = [0.7, 1.3] 
    # range of aspect ratio of the origin aspect ratio cropped
    rotation_prob = 1.0
    rotation_degrees = [-180, 180]
    translation_prob =  1
    translation_range =  [0.2, 0.2]
    # randomly convert image to grayscale
    grayscale_prob =  0.5
    # only available for torch version >= 1.7.1.
    gaussian_blur_prob = 0.2
    gaussian_blur_kernel_size= 7
    gaussian_blur_sigma= 0.5

    value_fill= 0


def data_transforms(main_cfg, aug_cfg):

    operations = {
        'random_crop': random_apply(
            transforms.RandomResizedCrop(
                size=(main_cfg.input_size[0], main_cfg.input_size[1]),
                scale=aug_cfg.random_crop_scale,
                ratio=aug_cfg.random_crop_ratio
            ),
            p= aug_cfg.random_crop_prob
        ),
        'horizontal_flip': transforms.RandomHorizontalFlip(
            p=aug_cfg.horizontal_flip_prob
        ),
        'vertical_flip': transforms.RandomVerticalFlip(
            p=aug_cfg.vertical_flip_prob
        ),
        'color_distortion': random_apply(
            transforms.ColorJitter(
                brightness=aug_cfg.brightness,
                contrast=aug_cfg.contrast,
                saturation=aug_cfg.saturation,
                hue=aug_cfg.hue
            ),
            p=aug_cfg.color_distortion_prob
        ),
        'rotation': random_apply(
            transforms.RandomRotation(
                degrees=aug_cfg.rotation_degrees,
                fill=aug_cfg.value_fill
            ),
            p=aug_cfg.rotation_prob
        ),
        'translation': random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=aug_cfg.translation_range,
                fillcolor=aug_cfg.value_fill
            ),
            p=aug_cfg.translation_prob
        ),
        'grayscale': transforms.RandomGrayscale(
            p=aug_cfg.grayscale_prob
        )
    }

    if version.parse(torch.__version__) >= version.parse('1.7.1'):
        operations['gaussian_blur'] = random_apply(
            transforms.GaussianBlur(
                kernel_size=aug_cfg.gaussian_blur_kernel_size,
                sigma=aug_cfg.gaussian_blur_sigma
            ),
            p=aug_cfg.gaussian_blur_prob
        )

    augmentations = []
    for op in main_cfg.data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.Resize((main_cfg.input_size[0], main_cfg.input_size[1])),
        transforms.ToTensor(),
        #transforms.Normalize(cfg.data.mean, cfg.data.std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    # test_preprocess = transforms.Compose(normalization)

    return train_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)


def simple_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.ToTensor()
    ])
