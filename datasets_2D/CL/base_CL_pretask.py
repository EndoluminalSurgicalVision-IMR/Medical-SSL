import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class CLBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.crop_size = config.input_size
        self._base_dir = base_dir
        self.all_images = []
        self.flag = flag

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass


