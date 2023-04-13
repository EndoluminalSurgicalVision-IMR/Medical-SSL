import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms
from datasets_2D.MG.eyepacs_mg_pretask import MGEyepacsPretaskSet
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

class PCRLEyepacsPretaskSet(MGEyepacsPretaskSet):
    def __init__(self, config, base_dir, flag):
        super(PCRLEyepacsPretaskSet, self).__init__(config, base_dir, flag)
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
        self.gauss_rate = 0.4
        self.Transforms_B = transforms.GaussianBlur(
                kernel_size=7,
                sigma=0.5
            )

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image_index = image_path[image_path.find('_1024') + 5:-4]

        if self.im_channel == 3:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        # Transforms Crop+Flip+Rotate
        input1 = self.aug_transform(image)
        input2 = self.aug_transform(image)

        input1 = input1.numpy()
        input2 = input2.numpy()

        # Get the gts and masks
        gt1 = copy.deepcopy(input1)
        gt2 = copy.deepcopy(input2)
        mask1 = copy.deepcopy(input1)
        mask2 = copy.deepcopy(input2)
        mask1, aug_tensor1 = self.spatial_aug(mask1)
        mask2, aug_tensor2 = self.spatial_aug(mask2)

        # Mix-up
        alpha = np.random.beta(1., 1.)
        alpha = max(alpha, 1 - alpha)
        input_h = alpha * gt1 + (1 - alpha) * gt2
        mask_h, aug_tensor_h = self.spatial_aug(input_h)

        # Transforms I+O+Blur
        if random.random() < self.paint_rate:
            if random.random() < self.inpaint_rate:
                # Inpainting
                input1 = self.image_in_painting(input1)
                input2 = self.image_in_painting(input2)
            else:
                # Outpainting
                input1 = self.image_out_painting(input1)
                input2 = self.image_out_painting(input2)

        input1 = torch.from_numpy(input1)
        input2 = torch.from_numpy(input2)
        if random.random() < self.gauss_rate:
            input1 = self.Transforms_B(input1)
            input2 = self.Transforms_B(input2)

        return input1, \
               input2, \
               torch.from_numpy(mask1), \
               torch.from_numpy(mask2), \
               torch.from_numpy(gt1), \
               torch.from_numpy(gt2), \
               torch.from_numpy(mask_h), \
               aug_tensor1, aug_tensor2, aug_tensor_h

    def spatial_aug(self, img):
        # img = img.numpy()
        c, h, w = img.shape
        aug_tensor = [0 for _ in range(6)]
        if random.random() < 0.5:
            img = np.flip(img, 1)
            aug_tensor[0] = 1
        if random.random() < 0.5:
            img = np.flip(img, 2)
            aug_tensor[1] = 1
        times = int(random.random() // 0.25)
        img = np.rot90(img, times, (1, 2))
        aug_tensor[times + 2] = 1
        return img.copy(), torch.tensor(aug_tensor)

