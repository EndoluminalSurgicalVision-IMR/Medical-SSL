import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class JigsawBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.base_dir = base_dir
        self.all_images = []
        self.flag = flag
        self.input_size = config.input_size
        # self.cube_size = config.cube_size
        self.gaps = config.gaps
        self.num_grids_per_axis = config.num_grids_per_axis
        self.num_cubes = self.num_grids_per_axis ** 2

        self.order_num_classes = config.order_class_num
        self.rot_num_classes = self.num_cubes

        self.K_permutations = np.load(config.k_permutations_path)
        assert self.order_num_classes == len(self.K_permutations)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

    def crop_2d(self, image, flag, crop_size):
        h, w = crop_size[0], crop_size[1]
        h_old, w_old = image.size()[1], image.size()[2]

        if flag == 'train':
            # crop random
            x = np.random.randint(0, 1 + h_old - h)
            y = np.random.randint(0, 1 + w_old - w)
        else:
            # crop center
            x = int((h_old - h) / 2)
            y = int((w_old - w) / 2)

        return self.do_crop_2d(image, x, y, h, w)

    def do_crop_2d(self, image, x, y, h, w):
        assert type(x) == int, x
        assert type(y) == int, y
        assert type(h) == int, h
        assert type(w) == int, w

        return image[:, x:x + h, y:y + w]

    def crop_cubes_2d(self, image, flag, cubes_per_side, cube_jitter_xy=3):
        c, h, w = image.shape

        patch_overlap = -cube_jitter_xy if cube_jitter_xy < 0 else 0

        h_grid = (h - patch_overlap) // cubes_per_side
        w_grid = (w - patch_overlap) // cubes_per_side

        h_patch = h_grid - cube_jitter_xy
        w_patch = w_grid - cube_jitter_xy

        cubes = torch.zeros([cubes_per_side ** 2, c, h_patch, w_patch])
        count = 0
        for i in range(cubes_per_side):
            for j in range(cubes_per_side):

                p = self.do_crop_2d(image,
                                    i * h_grid,
                                    j * w_grid,
                                    h_grid + patch_overlap,
                                    w_grid + patch_overlap)

                if h_patch < h_grid or w_patch < w_grid:
                    p = self.crop_2d(p, flag, [h_patch, w_patch])

                cubes[count] = p
                count += 1

        return cubes

    def rearrange(self, cubes, K_permutations):
        label = random.randint(0, len(K_permutations) - 1)
        return cubes[K_permutations[label]], label

