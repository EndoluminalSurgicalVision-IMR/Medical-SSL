import numpy as np
from tqdm import tqdm
import os

import torch
import torchio.transforms
from .base_ptp_pretask import PTPBase
import glob


class RotLunaPretaskSet(PTPBase):
    """
       Luna Dataset for SSM-rotation prediction (ROT).
        https://proceedings.neurips.cc/paper/2020/file/d2dc6368837861b42020ee72b0896182-Paper.pdf
    """
    def __init__(self, config, base_dir, flag):
        super(RotLunaPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.org_data_size
        self.input_size = config.input_size
        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        # torchio input dim [C, H, W, D]
        self.resize = torchio.transforms.Resize(self.input_size)
        self.num_rotations_per_patch = config.num_rotations_per_patch

        # load data
        self.get_luna_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"
        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image_path = self.all_images[index]
        image_array = np.load(image_path)
        image_array = np.expand_dims(image_array, 0)
        input = self.random_crop(image_array, self.input_size)

        # [128， 128， 64] -> [128， 128， 128]
        input = self.resize(input)
        rotation_labels = np.random.choice(range(0, 10), self.num_rotations_per_patch)

        # generate multiple rotate versions for each 3D volume.
        if self.num_rotations_per_patch > 1:
            rotated_imgs = []
            for v in range(self.num_rotations_per_patch):
                rotated_imgs.append(self.rotate_3d(input, rotation_labels[v]))
            rotated_input = np.stack(rotated_imgs, 0)
            # rotated_input [4, c, h, w, d]
        else:
            # num_rotations_per_patch = 1
            rotated_input = self.rotate_3d(input, rotation_labels[0])
            # rotated_input [c, h, w, d]

        return torch.from_numpy(rotated_input).float(), torch.from_numpy(np.array(rotation_labels)).long()

    def get_luna_list(self):
        self.all_images = []
        for index_subset in self.folds:
            luna_subset_path = os.path.join(self.base_dir, "subset" + str(index_subset))
            file_list = glob.glob(os.path.join(luna_subset_path, "*.npy"))
            # save the paths
            for img_file in tqdm(file_list):
                self.all_images.append(img_file)
        # x_train: (14159)
        # x_valid: (5640)
        return


class RotLunaPretaskSet_v2(PTPBase):
    """
       Luna Dataset for SSM-rotation prediction (ROT).
        From https://github.com/HealthML/self-supervised-3d-tasks/blob/master/self_supervised_3d_tasks
    """
    def __init__(self, config, base_dir, flag):
        super(RotLunaPretaskSet_v2, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.org_data_size
        self.input_size = config.input_size
        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        # torchio input dim [C, W, H, D]
        self.resize = torchio.transforms.Resize(self.input_size)
        self.num_rotations_per_patch = config.num_rotations_per_patch

        # load data
        self.get_luna_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image_array = np.load(image_path)
        image_array = np.expand_dims(image_array, 0)
        input = self.random_crop(image_array, self.crop_size)

        # [128， 128， 64] -> [128， 128， 128]
        input = self.resize(input)

        # generate multiple rotate versions for each 3D volume.
        if self.num_rotations_per_patch > 1:
            rotated_imgs = []
            rotated_labels = []
            for i in range(4):
                rot_img, rot_label = self.rotate_3d_v2(input)
                rotated_imgs.append(rot_img)
                rotated_labels.append(rot_label)
            # rotated_input [4, c, h, w, d]
        else:
            # num_rotations_per_patch = 1
            rotated_imgs, rotated_labels = self.rotate_3d_v2(input)
            # rotated_input [c, h, w, d]

        rotated_input = np.array(rotated_imgs)
        rotated_labels = np.array(rotated_labels)

        return torch.from_numpy(rotated_input).float(), torch.from_numpy(rotated_labels).long()

    def get_luna_list(self):
        self.all_images = []
        for index_subset in self.folds:
            luna_subset_path = os.path.join(self.base_dir, "subset" + str(index_subset))
            file_list = glob.glob(os.path.join(luna_subset_path, "*.npy"))
            # save the paths
            for img_file in tqdm(file_list):
                self.all_images.append(img_file)
        # x_train: (14159)
        # x_valid: (5640)
        return
