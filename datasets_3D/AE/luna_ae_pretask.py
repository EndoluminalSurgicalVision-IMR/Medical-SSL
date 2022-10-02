import copy
import numpy as np
import torch
from tqdm import tqdm
import os

from .base_ae_pretask import AEBase


class AELunaPretaskSet(AEBase):
    """
       Luna Dataset for AE.
       """
    def __init__(self, config, base_dir, flag):
        super(AELunaPretaskSet, self).__init__(config, base_dir, flag)
        self.crop_size = config.input_size
        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold
        self.flip_rate = 0.4

        # load data from pre-saved files ".npy"
        self.get_luna_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        input = self.all_images[index]

        # Autoencoder
        gt = copy.deepcopy(input)

        # Flipping
        input, gt = self.data_augmentation(input, gt, self.flip_rate)

        return torch.from_numpy(input.copy()).float(), torch.from_numpy(gt.copy()).float()

    def get_luna_list(self):
        self.all_images = []

        for i, fold in enumerate(tqdm(self.folds)):
            file_name = "bat_" + str(self.config.scale) + "_s_" + str(self.crop_size[0]) + "x" + str(
                self.crop_size[1]) + "x" + str(
                self.crop_size[2]) + "_" + str(fold) + ".npy"
            print('***file_name**:', file_name)
            s = np.load(os.path.join(self.base_dir, file_name))
            self.all_images.extend(s)
        self.all_images = np.expand_dims(np.array(self.all_images), axis=1)

        print("x_{}: {} | {:.2f} ~ {:.2f}".format(self.flag, self.all_images.shape, np.min(self.all_images), np.max(self.all_images)))
        # x_train: (14159, 1, 64, 64, 32) | 0.00 ~ 1.00
        # x_valid: (5640, 1, 64, 64, 32) | 0.00 ~ 1.00
        return



