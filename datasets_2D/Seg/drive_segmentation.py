import torchvision.transforms.functional as tf
import torch
from torchvision import transforms
import numpy as np
from scipy import ndimage
import random
import os
import csv
from datasets_2D.paths import Path
from datasets_2D.Seg.base_segmentation import SegmentationBase
import argparse
from torch.utils.data import DataLoader
import utils.tools as utils
from PIL import Image
from torchvision.transforms import transforms, ToTensor
from packaging import version
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class SegDRIVESet(SegmentationBase):
    """
   dataset for segmentation.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/valid/test
        """
        super(SegDRIVESet, self).__init__(config, base_dir, flag)

        self.all_images = []
        self.all_labels = []
        if self.flag == 'train':
            self.root_dir = os.path.join(base_dir , 'train')
        elif self.flag == 'test':
            self.root_dir = os.path.join(base_dir , 'test')
        else:
            self.root_dir = os.path.join(base_dir , 'valid')

        self.all_images = os.listdir(os.path.join(self.root_dir, 'images'))
      
        assert len(self.all_images) != 0, "the images can`t be zero!"
        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

        self.transform = ToTensor()

    def __getitem__(self, index):
        image_path = self.root_dir + '/images_eh_in_mask/' + self.all_images[index]
        name =  self.all_images[index][:2]
        mask_path = self.root_dir + '/masks/' + self.all_images[index][:-4] + '_mask.gif'
        label_path = self.root_dir + '/labels/' + name + '_manual1.gif'
       
        if self.im_channel == 3:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image_tensor = self.transform(image)
        label_tensor = torch.from_numpy((np.array(label)/255)).squeeze().unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask)/255).squeeze().unsqueeze(0)

        image_tensor, label_tensor, mask_tensor = self.check_size(image_tensor, label_tensor, mask_tensor)

        return image_tensor, label_tensor ,mask_tensor, name

    def check_size(self, img_tensor, label_tensor, mask_tensor):
        c, h, w = img_tensor.size()
        scale=16
        if h//scale !=0:
            img_tensor = img_tensor[:, 0:(h // scale) * scale, :]
            label_tensor = label_tensor[:, 0:(h // scale) * scale, :]
            mask_tensor = mask_tensor[:, 0:(h // scale) * scale, :]
        if w // scale != 0:
            img_tensor = img_tensor[:, :, 0:(w // scale) *scale]
            label_tensor = label_tensor[:, :, 0:(w // scale) *scale]
            mask_tensor = mask_tensor[:, :, 0:(w // scale) * scale]

        return img_tensor, label_tensor, mask_tensor

