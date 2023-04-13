import copy
import random
import time
import os
from glob import glob

import numpy as np
from scipy.special import comb
import torch
import torchio.transforms
import csv
from datasets_2D.CL.base_CL_pretask import CLBase
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import argparse
from torch.utils.data import DataLoader
from utils.tools import save_tensor2image

from tqdm import tqdm

DATA_CONFIG = {
    # 'input_size': 128,
    # 'patch_size': 128,
    'data_augmentation': {
        'brightness': 0.4,  # how much to jitter brightness
        'contrast': 0.4,  # How much to jitter contrast
        'saturation': 0.4,
        'hue': 0.1,
        'scale': (0.8, 1.2),  # range of size of the origin size cropped
        'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
        'degrees': (-180, 180),  # range of degrees to select from
        'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
    }
}


class CLEyepacsPretaskSet(CLBase):
    def __init__(self, config, base_dir, flag):
        super(CLEyepacsPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self._base_dir = base_dir
        self.ratio = self.config.ratio
        self.flag = flag
        self.input_size = config.input_size
        self.im_channel = config.im_channel
        self.all_images = []
        if self.flag == 'train':
            self.root_dir = os.path.join(self._base_dir, flag + 'train_1024') 
        else:
            self.root_dir =os.path.join(self._base_dir, flag + 'test_1024')

        with open(os.path.join(self._base_dir, flag + '.csv')) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            for row in reader:
                self.all_images.append(self.root_dir + '/' + row[1] + '.jpg')

            f.close()

        assert len(self.all_images) != 0, "the images can`t be zero!"

        if self.flag == 'train':
            self.all_images = self.all_images[:int(self.ratio * len(self.all_images))]
        else:
            self.all_images = self.all_images[:1000]

        data_aug = DATA_CONFIG['data_augmentation']

        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=data_aug['brightness'],
                contrast=data_aug['contrast'],
                saturation=data_aug['saturation'],
                hue=data_aug['hue']
            ),
            transforms.RandomResizedCrop(
                size=(self.input_size[0], self.input_size[1]),
                scale=data_aug['scale'],
                ratio=data_aug['ratio']
            ),
            transforms.RandomAffine(
                degrees=data_aug['degrees'],
                translate=data_aug['translate']
            ),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
        ])

        ### Display status
        print('Number of images in {}: {:d},  Ratio: {}'.format(flag, len(self.all_images), self.ratio))


    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        if self.im_channel == 3:
            image = Image.open(self.all_images[index]).convert('RGB')
        else:
            image = Image.open(self.all_images[index]).convert('L')

        input1 = self.aug_transform(image)
        input2 = self.aug_transform(image)

        return input1.float(), input2.float()

