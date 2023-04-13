from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import torch
from torchvision import transforms
import numpy as np
from scipy import ndimage
import random
import os
import cv2
from PIL import Image
from torchvision.transforms import transforms, ToTensor


class ClassificationBase(Dataset):
    """
   Base_dataset for classification.
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

        super(ClassificationBase, self).__init__()
        self.flag = flag
        self.config = config
        self.num_classes = config.class_num
        self.im_channel = config.im_channel
        self._base_dir = base_dir
        self.all_images = []
        self.all_labels = []
        self.root_dir = None
        self.input_size = config.input_size[0]
        if self.input_size < 1024:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size, interpolation=Image.ANTIALIAS), 
                ToTensor()])
        else:
            self.transform = transforms.Compose([
                ToTensor()])

    def __len__(self):
            return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        if self.im_channel == 3:
            image = Image.open(self.all_images[index]).convert('RGB')
        else:
            image = Image.open(self.all_images[index]).convert('L')
        image_tensor = self.transform(image)
        label = np.array(self.all_labels[index]).astype(np.int32)
        label = torch.from_numpy(label).long()
        return image_tensor, label, image_path



