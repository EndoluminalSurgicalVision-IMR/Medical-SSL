
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

# SSM: Relative 2D patch location (2D-RPL)

class RPLEyepacsPretaskSet(PTPBase):
    def __init__(self, config, base_dir, flag):
        super(RPLEyepacsPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.input_size = config.input_size
        self.center_size = 720
        self.patch_size = (self.center_size//2) // config.num_grids_per_axis - 6
        # N (num*num) patches in an image, N-1 way classification
        self.num_grids_per_axis = config.num_grids_per_axis
        assert self.num_grids_per_axis ** 2 == config.class_num + 1
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

        self.transform = transforms.Compose([
            transforms.CenterCrop((720, 720)), 
            transforms.Resize((720//2, 720//2)),
            ToTensor()])

        if self.flag == 'train':
            self.all_images = self.all_images[:int(self.ratio * len(self.all_images))]
        else:
            self.all_images = self.all_images[:3000]
           
        ### Display status
        print('Number of images in {}: {:d},  Ratio: {}'.format(flag, len(self.all_images), self.ratio))


    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        # image_index = image_path[image_path.find('_1024') + 5:-4]

        if self.im_channel == 3:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        image_tensor = self.transform(image)
        uniform_patch, random_patch, label = self.get_patch_from_grid_tensor(image_tensor, patch_dim=self.patch_size, gap=3)

        return uniform_patch, random_patch,\
               torch.from_numpy(np.array(label))

    def get_patch_from_grid_tensor(self, image, patch_dim, gap):

        offset_x, offset_y = image.shape[1] - (patch_dim * 3 + gap * 2), image.shape[2] - (patch_dim * 3 + gap * 2)
        start_grid_x, start_grid_y = np.random.randint(0, offset_x), np.random.randint(0, offset_y)
        patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
        loc = np.random.randint(len(patch_loc_arr))

        tempx, tempy = patch_loc_arr[loc]
       

        patch_x_pt = start_grid_x + patch_dim * (tempx - 1) + gap * (tempx - 1)
        patch_y_pt = start_grid_y + patch_dim * (tempy - 1) + gap * (tempy - 1)
        random_patch = image[:, patch_x_pt:patch_x_pt + patch_dim, patch_y_pt:patch_y_pt + patch_dim]

        patch_x_pt = start_grid_x + patch_dim * (2 - 1) + gap * (2 - 1)
        patch_y_pt = start_grid_y + patch_dim * (2 - 1) + gap * (2 - 1)
        uniform_patch = image[:, patch_x_pt:patch_x_pt + patch_dim, patch_y_pt:patch_y_pt + patch_dim]

        random_patch_label = loc

        return uniform_patch, random_patch, random_patch_label
