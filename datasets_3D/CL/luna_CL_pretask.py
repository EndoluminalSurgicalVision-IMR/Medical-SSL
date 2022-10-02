import os
from glob import glob

import numpy as np
import torch
import torchio.transforms
from datasets_3D.CL.base_CL_pretask import CLBase


class CLLunaPretaskSet(CLBase):
    """
    Luna Dataset for contrastive learning-based SSL methods.
    """
    def __init__(self, config, base_dir, flag):
        super(CLLunaPretaskSet, self).__init__(config, base_dir, flag)
        self.ratio = self.config.ratio
        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        self.all_images = self.get_file_list()

        # self.DEFAULT_AUG = torchio.transforms.Compose([
        #                      torchio.transforms.RandomFlip(),
        #                      # torchio.transforms.Affine(
        #                      #        scales=tuple((0.9, 1.2, 0.9, 1.2, 1, 1)),
        #                      #        degrees=tuple((10, -10, 10, -10, 0, 0)),
        #                      #        translation=tuple((-2, 2, -2, 2, 0, 0))),
        #                      torchio.transforms.RandomBlur()
        #                      ])
        self.DEFAULT_AUG = torchio.transforms.Compose([
                           torchio.transforms.RandomFlip(),
                           torchio.transforms.RandomAffine(scales=(0.8, 1.2, 0.8, 1.2, 1, 1),
                                                           degrees=(-10, 10, -10, 10, 0, 0)),
                           torchio.transforms.RandomBlur()
                           ])

        assert len(self.all_images) != 0, "the images can`t be zero!"

        # Display status
        print('Number of images in {}: {:d}, Fold-Index:{}, Ratio: {}'.format(flag, len(self.all_images), self.folds, self.ratio))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_name = self.all_images[index]
        pair = np.load(image_name)

        # The pairs were randomly cropped and resized with iou > 0.25
        crop1 = pair[0]
        crop1 = np.expand_dims(crop1, axis=0)
        crop2 = pair[1]
        crop2 = np.expand_dims(crop2, axis=0)

        # augmentation by DEFAULT_AUG
        input1 = self.DEFAULT_AUG(crop1)
        input2 = self.DEFAULT_AUG(crop2)
        return torch.from_numpy(input1).float(), torch.from_numpy(input2).float()

    def get_file_list(self):
        all_images = []
        for i in self.folds:
            subset = os.path.join(self._base_dir, 'subset' + str(i))
            for file in glob(os.path.join(subset, '*.npy')):
                all_images.append(file)
        return all_images[:int(len(all_images) * self.ratio)]

