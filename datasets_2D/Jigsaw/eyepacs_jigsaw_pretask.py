import copy
import random
import numpy as np
import torch
import torchio.transforms
from tqdm import tqdm
import os
import glob
import SimpleITK as sitk
from scipy import ndimage
from utils.tools import save_tensor2image
from datasets_2D.Jigsaw.base_jigsaw_pretask import JigsawBase
import csv
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import argparse
from torch.utils.data import DataLoader


class JigSawEyepacsPretaskSet(JigsawBase):
    def __init__(self, config, base_dir, flag):
        super(JigSawEyepacsPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        # self.input_size = config.input_size
        # self.cube_size = config.cube_size
        self.ratio = config.ratio
        self.im_channel = config.im_channel

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
        else:
            self.all_images = self.all_images[:3000]

        # self.transform = transforms.Compose([
        #     transforms.CenterCrop((720, 720)),
        #     transforms.Resize((self.input_size[0], self.input_size[1])),
        #     ToTensor()])
        self.transform = transforms.Compose([
            transforms.CenterCrop((720, 720)),  # 682
            transforms.Resize((720 // 2, 720 // 2)),
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

        # [C, H, W]
        # input = image_tensor.numpy()

        # get all the num_grids **2 cubes
        all_cubes = self.crop_cubes_2d(image_tensor,
                                       flag=self.flag,
                                       cubes_per_side=self.num_grids_per_axis,
                                       cube_jitter_xy=6)

        # print('cubes', all_cubes.size(), all_cubes.min(), all_cubes.max())

        # # Task1: Permutate the order of cubes
        rearranged_cubes, order_label = self.rearrange(all_cubes, self.K_permutations)

        return rearranged_cubes, torch.from_numpy(np.array(order_label)) 




