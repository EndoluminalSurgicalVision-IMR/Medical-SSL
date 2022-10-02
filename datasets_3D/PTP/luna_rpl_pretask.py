
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
from .base_ptp_pretask import PTPBase
import glob


class RPLLunaPretaskSet(PTPBase):
    """
      Luna Dataset for SSM-relative positive Localization (RPL).
       https://proceedings.neurips.cc/paper/2020/file/d2dc6368837861b42020ee72b0896182-Paper.pdf
    """
    def __init__(self, config, base_dir, flag):
        super(RPLLunaPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.input_size

        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        # N (num*num*num) patches in a volume, N-1 way classification
        self.num_grids_per_axis = config.num_grids_per_axis
        assert self.num_grids_per_axis ** 3 == config.class_num + 1

        # self.resize = torchio.transforms.Resize([320, 320, self.len_z])

        # load data
        self.get_luna_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image_path = self.all_images[index]
        input = np.load(image_path)

        center_patch, random_patch, label = self.get_patch_from_grid(input, gap=3)
        if len(input.shape) < 4:
            center_patch = np.expand_dims(center_patch, 0)
            random_patch = np.expand_dims(random_patch, 0)

        return torch.from_numpy(center_patch.astype(np.float32)),\
               torch.from_numpy(random_patch.astype(np.float32)), \
               torch.from_numpy(label).long()

    def get_patch_from_grid(self, image, gap):
        """
        3D version based on the 2D version in https://github.com/abhisheksambyal/Self-supervised-learning-by-context-prediction/blob/master/Self_supervised_learning_by_context_prediction.ipynb
        image: [C, X, Y, Z] or [X, Y, Z]
        """


        offset_x, offset_y, offset_z = image.shape[-3] - (self.crop_size[0] * 3 + gap * 2),\
                                       image.shape[-2] - (self.crop_size[1] * 3 + gap * 2),\
                                       image.shape[-1] - (self.crop_size[2] * 3 + gap * 2)

        start_grid_x, start_grid_y, start_grid_z = np.random.randint(0, offset_x),\
                                                   np.random.randint(0, offset_y),\
                                                   np.random.randint(0, offset_z)

        # patch_loc_arr = [(1, 1, 1), (1, 1, 2), (1, 1, 3),
        #                  (2, 1, 1), (2, 1, 2), (2, 1, 3),
        #                  (3, 1, 1), (3, 1, 2), (3, 1, 3),
        #                  (1, 2, 1), (1, 2, 2), (1, 2, 3),
        #                  (2, 2, 1),            (2, 2, 3),
        #                  (3, 2, 1), (3, 2, 2), (3, 2, 3),
        #                  (1, 3, 1), (1, 3, 2), (1, 3, 3),
        #                  (2, 3, 1), (2, 3, 2), (2, 3, 3),
        #                  (3, 3, 1), (3, 3, 2), (3, 3, 3)
        #                  ]

        patch_loc_arr = []

        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(1, 4):
                    if i == 2 and j == 2 and k == 2:
                        continue
                    patch_loc_arr.append((i, j, k))

        assert len(patch_loc_arr) == 26

        # randint [low, high)
        loc = np.random.randint(len(patch_loc_arr))
        tempx, tempy, tempz = patch_loc_arr[loc]

        patch_x_pt = start_grid_x + self.crop_size[0] * (tempx - 1) + gap * (tempx - 1)
        patch_y_pt = start_grid_y + self.crop_size[1] * (tempy - 1) + gap * (tempy - 1)
        patch_z_pt = start_grid_z + self.crop_size[2] * (tempz - 1) + gap * (tempz - 1)

        if len(image.shape) == 3:
            # [x, y, z]
            random_patch = image[patch_x_pt:patch_x_pt + self.crop_size[0],
                                 patch_y_pt:patch_y_pt + self.crop_size[1],
                                 patch_z_pt:patch_z_pt + self.crop_size[2]]
            assert random_patch.shape == (self.crop_size[0], self.crop_size[1], self.crop_size[2])
        else:
            # [c, x, y, z]
            random_patch = image[:, patch_x_pt:patch_x_pt + self.crop_size[0],
                           patch_y_pt:patch_y_pt + self.crop_size[1],
                           patch_z_pt:patch_z_pt + self.crop_size[2]]

            assert random_patch.shape == (1, self.crop_size[0], self.crop_size[1], self.crop_size[2])

        # (2, 2, 2) center patch
        patch_x_pt = start_grid_x + self.crop_size[0] * (2 - 1) + gap * (2 - 1)
        patch_y_pt = start_grid_y + self.crop_size[1] * (2 - 1) + gap * (2 - 1)
        patch_z_pt = start_grid_z + self.crop_size[2] * (2 - 1) + gap * (2 - 1)

        if len(image.shape) == 3:
            # [x, y, z]
            center_patch = image[patch_x_pt:patch_x_pt + self.crop_size[0],
                                 patch_y_pt:patch_y_pt + self.crop_size[1],
                                 patch_z_pt:patch_z_pt + self.crop_size[2]]
            assert center_patch.shape == (self.crop_size[0], self.crop_size[1], self.crop_size[2])
        else:
            #[c, x, y, z]
            center_patch = image[:, patch_x_pt:patch_x_pt + self.crop_size[0],
                           patch_y_pt:patch_y_pt + self.crop_size[1],
                           patch_z_pt:patch_z_pt + self.crop_size[2]]

            assert center_patch.shape == (1, self.crop_size[0], self.crop_size[1], self.crop_size[2])

        random_patch_label = loc

        return center_patch, random_patch, np.array(random_patch_label)

    def get_luna_list(self):
        self.all_images = []
        for index_subset in self.folds:
            luna_subset_path = os.path.join(self.base_dir, "subset" + str(index_subset))
            file_list = glob.glob(os.path.join(luna_subset_path, "*.npy"))

            for img_file in tqdm(file_list):
                self.all_images.append(img_file)
        return


class RPLLunaPretaskSet_v2(PTPBase):
    """
     Luna Dataset for SSM-relative positive Localization (RPL).
      From https://github.com/HealthML/self-supervised-3d-tasks/blob/master/self_supervised_3d_tasks/preprocessing/preprocess_rpl.py
    """
    def __init__(self, config, base_dir, flag):
        super(RPLLunaPretaskSet_v2, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.input_size
        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        # N (num*num*num) patches in a volume, N-1 way classification
        self.num_grids_per_axis = config.num_grids_per_axis
        assert self.num_grids_per_axis ** 3 == config.class_num + 1

        #self.resize = torchio.transforms.Resize([320, 320, self.len_z])

        # load data
        self.get_luna_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image_path = self.all_images[index]
        input = np.load(image_path)

        center_patch, random_patch, label = self.get_patch_from_grid(input,
                                                                     num_grids_per_axis=self.num_grids_per_axis,
                                                                     patch_jitter=8)
        if len(input.shape) < 4:
            center_patch = np.expand_dims(center_patch, 0)
            random_patch = np.expand_dims(random_patch, 0)


        return torch.from_numpy(center_patch.astype(np.float32)),\
               torch.from_numpy(random_patch.astype(np.float32)), \
               torch.from_numpy(label).long()


    def get_patch_from_grid(self, img, num_grids_per_axis, patch_jitter=8):
        patch_count = num_grids_per_axis ** 3
        center_id = int(patch_count / 2)

        cropped_pathes = self.crop_patches_3d(img,
                                              num_grids_per_axis,
                                              patch_jitter_xy=patch_jitter,
                                              patch_jitter_z=patch_jitter)

        class_id = np.random.randint(patch_count - 1)
        patch_id = class_id
        if class_id >= center_id:
            patch_id = class_id + 1

        return cropped_pathes[center_id], cropped_pathes[patch_id], np.array(class_id)

    def get_luna_list(self):
        self.all_images = []
        for index_subset in self.folds:
            luna_subset_path = os.path.join(self.base_dir, "subset" + str(index_subset))
            file_list = glob.glob(os.path.join(luna_subset_path, "*.npy"))

            for img_file in tqdm(file_list):
                self.all_images.append(img_file)
        return


