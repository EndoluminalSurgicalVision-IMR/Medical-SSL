import copy
import random
import numpy as np
import torch
import torchio.transforms
from tqdm import tqdm
import os
import csv
from datasets_2D.MG.base_mg_pretask import MGBase2D
import cv2
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import argparse
from torch.utils.data import DataLoader
from utils.tools import save_tensor2image
from datasets_2D.MG.eyepacs_mg_pretask import MGEyepacsPretaskSet


class AEEyepacsPretaskSet(MGEyepacsPretaskSet):
    def __init__(self, config, base_dir, flag):
        super(AEEyepacsPretaskSet, self).__init__(config, base_dir, flag)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image_index = image_path[image_path.find('_1024')+5:-4]

        if self.im_channel == 3:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')

        image_tensor = self.transform(image)

        # [C, H, W]
        input = image_tensor.numpy()

        # Autoencoder
        gt = copy.deepcopy(input)

        # Flip
        input, gt = self.data_augmentation(input, gt, self.flip_rate)


        return  torch.from_numpy(input.copy()).float(), torch.from_numpy(gt.copy()).float(), image_index







