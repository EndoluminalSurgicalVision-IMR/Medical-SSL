# coding: utf-8

"""
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /mnt/dataset/shared/zongwei/LUNA16 \
--save generated_cubes
done
"""

# In[1]:


import warnings

warnings.filterwarnings('ignore')
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize
from utils.tools import save_np2nii

sys.setrecursionlimit(40000)


def generator_from_one_volume(img_array, org_data_size=[320, 320, 74]):
    size_x, size_y, size_z = img_array.shape

    hu_min = -1000.
    hu_max = 1000.
    img_array[img_array < hu_min] = hu_min
    img_array[img_array > hu_max] = hu_max
    img_array = 1.0 * (img_array - hu_min) / (hu_max - hu_min)

    h1 = int(round((size_x - org_data_size[0]) / 2.))
    w1 = int(round((size_y - org_data_size[1]) / 2.))
    d1 = int(round((size_z - org_data_size[2]) / 2.))

    img_array = img_array[h1:h1 + org_data_size[0], w1:w1 + org_data_size[1], d1:d1 + org_data_size[2]]

    print(img_array.shape)

    return img_array


def get_self_learning_data(fold, data_path):
    for index_subset in fold:
        luna_subset_path = os.path.join(data_path, "subset" + str(index_subset))
        file_list = glob(os.path.join(luna_subset_path, "*.mhd"))
        print('{} files in fold {}'.format(len(file_list), index_subset))
        for img_file in tqdm(file_list):
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)
            img_array = img_array.transpose(2, 1, 0)

            x = generator_from_one_volume(img_array, org_data_size=[320, 320, 74])
            # print(os.path.split(img_file)[1][:-4])
            save_path = '../../Data/LUNA2016_cropped_x320y320z74/' + "subset" + str(index_subset)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path+ '/' + str(os.path.split(img_file)[1][:-4]) + '.npy', x)
            print("cube: {} | {:.2f} ~ {:.2f}".format(x.shape, np.min(x), np.max(x)))


for fold in [0, 1, 2, 3, 4, 5, 6]:
    print(">> Fold {}".format(fold))
    get_self_learning_data([fold], data_path='../../Data/LUNA2016')

    #np.save(os.path.join('../../Data/LUNA2016_croppped', str(fold) + ".npy"), cube)

#