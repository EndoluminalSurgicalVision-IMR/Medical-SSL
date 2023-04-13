
import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class PTPBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.base_dir = base_dir
        self.all_images = []
        self.flag = flag
        self.input_size = config.input_size
        self.im_channel = config.im_channel
        self.ratio = config.ratio


    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

    def random_crop(self, image, crop_size):
        """Crop the ndimage in a sample randomly.
              Args:
                  image:[C, H, W]
                  crop_size: the desired output size: [s0, s1]
                  out_image:[C, s0, s1]
               """
        _, h, w = image.shape
        if np.random.uniform() > 0.33:
            h1 = np.random.randint((h - crop_size[0]) // 4, 3 * (h - crop_size[0]) // 4)
            w1 = np.random.randint((w - crop_size[1]) // 4, 3 * (w - crop_size[1]) // 4)

        else:
            h1 = np.random.randint(0, h - crop_size[0])
            w1 = np.random.randint(0, w - crop_size[1])

        image = image[:, h1:h1 + crop_size[0], w1:w1 + crop_size[1]]

        return image

    def center_crop(self, image, size):
        """CenterCrop a ndimage.
           Args:
              image: [C, H, W]

              crop_size: the desired output size:
              out_image:[C, D, size, size]
              out_label:[K, D, size, size]
        """
        _, h, w = image.shape

        w1 = int(round((w - size) / 2.))
        h1 = int(round((h - size) / 2.))

        image = image[:, h1:h1 + size, w1:w1 + size]
        return image


    def rotate_ndimg(self, img):
        ### For Rot [H, W, C]
        #  classes: 0:3
        rot_class = np.random.choice(range(4))
        if rot_class != 0:
            img = np.rot90(img, k=rot_class)

        return img.copy(), rot_class

    def rotate_tensor(self, img):
        ### For Rot [C, H, W, C]
        #  classes: 0:3
        rot_class = np.random.choice(range(4))
        if rot_class != 0:
            img = torch.rot90(img, rot_class, (1, 2))
        return img, rot_class



