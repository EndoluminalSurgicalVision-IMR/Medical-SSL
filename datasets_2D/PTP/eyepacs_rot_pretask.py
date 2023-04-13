import copy
import random
import time

import numpy as np
from tqdm import tqdm
import os
import torch
from PIL import Image
from scipy.special import comb
import torchio.transforms
import csv
from PIL import Image
from torchvision.transforms import transforms, ToTensor
from datasets_2D.PTP.base_ptp_pretask import PTPBase
import argparse
from utils.tools import save_tensor2image
from torch.utils.data import DataLoader


# SSM: 2D Rotation prediction (2D-Rot)


class RotEyepacsPretaskSet(PTPBase):
    def __init__(self, config, base_dir, flag):
        super(RotEyepacsPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.input_size
        self.all_images = []
        if self.flag == 'train':
            self.root_dir = os.path.join(self.base_dir, 'train_1024')
        else:
            self.root_dir = os.path.join(self.base_dir, 'test_1024')

        with open(os.path.join(self.base_dir, flag + '.csv')) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            for row in reader:
                self.all_images.append(self.root_dir + '/' + row[1] + '.jpg')

            f.close()

        assert len(self.all_images) != 0, "the images can`t be zero!"

        if self.flag == 'train':
            self.all_images = self.all_images[:int(self.ratio * len(self.all_images))]
            self.transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomResizedCrop(
                    size=(self.input_size[0], self.input_size[1]),
                    scale=[0.85, 1.15],
                    ratio=[0.8, 1.2])], p=0.5),
                # transforms.RandomApply([transforms.RandomAffine(
                # degrees=0,
                # translate=[0.2, 0.2],
                # fillcolor=0
                # )], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0,
                    hue=0)], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0,
                    hue=0)], p=0.5),
                transforms.Resize((self.input_size[0], self.input_size[1])),
                transforms.ToTensor()
            ])
        else:
            self.all_images = self.all_images[:3000]
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size[0], self.input_size[1])),
                ToTensor()])
            ### Display status
        print('Number of images in {}: {:d},  Ratio: {}'.format(flag, len(self.all_images), self.ratio))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image_path = self.all_images[index]
        image_index = image_path[image_path.find('_1024') + 5:-4]

        if self.im_channel == 3:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        image_tensor = self.transform(image)

        rotated_input, label = self.rotate_tensor(image_tensor)

        return rotated_input, torch.from_numpy(np.array(label))
