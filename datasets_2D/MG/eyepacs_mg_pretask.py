import copy
import random
import numpy as np
import torch
import torchio.transforms
from tqdm import tqdm
import os
import csv
from datasets_2D.MG.base_mg_pretask import MGBase2D
import cv2
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import argparse
from torch.utils.data import DataLoader
from utils.tools import save_tensor2image


class MGEyepacsPretaskSet(MGBase2D):
    def __init__(self, config, base_dir, flag):
        super(MGEyepacsPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.input_size = config.input_size
        self.im_channel = config.im_channel
        self.ratio = config.ratio
        # image deformation
        self.nonlinear_rate = config.nonlinear_rate
        self.paint_rate = config.paint_rate
        self.outpaint_rate = config.outpaint_rate
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.local_rate = config.local_rate
        self.flip_rate = config.flip_rate
        # load data
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
                # transforms.RandomApply([transforms.RandomResizedCrop(
                #     size=(self.input_size[0], self.input_size[1]),
                #     scale=[0.87, 1.15],
                #     ratio=[0.7, 1.3])], p=0.5),
                transforms.Resize((self.input_size[0], self.input_size[1])),
                transforms.ToTensor()
            ])
        else:
            self.all_images = self.all_images[:500]
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size[0], self.input_size[1])),
                ToTensor()])
        ### Display status
        print('Number of images in {}: {:d},  Ratio: {}'.format(flag, len(self.all_images), self.ratio))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image_index = image_path[image_path.find('_1024')+5:-4]

        if self.im_channel == 3:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        image_tensor = self.transform(image)

        # [C, H, W]
        input = image_tensor.numpy()

        # Autoencoder
        gt = copy.deepcopy(input)

        # Flip
        input, gt = self.data_augmentation(input, gt, self.flip_rate)

        # Local Shuffle Pixel
        input = self.local_pixel_shuffling(input, prob=self.local_rate)

        # Apply non-Linear transformation with an assigned probability
        input = self.nonlinear_transformation(input, self.nonlinear_rate)

        # Inpainting & Outpainting
        if random.random() < self.paint_rate:
            if random.random() < self.inpaint_rate:
                # Inpainting
                input = self.image_in_painting(input)
            else:
                # Outpainting
                input = self.image_out_painting(input)


        return  torch.from_numpy(input.copy()).float(), torch.from_numpy(gt.copy()).float(), image_index


