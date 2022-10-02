# coding: utf-8
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time
from glob import glob

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

import copy
import skimage.measure as measure
import skimage.morphology as morphology

from utils.tools import save_np2nii
import cv2
import csv


def clahe_equalized(imgs):
   assert (len(imgs.shape)==3)  #3D arrays
   #create a CLAHE object (Arguments are optional).
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   imgs_equalized = np.empty(imgs.shape)
   for i in range(imgs_equalized.shape[0]):
       imgs_equalized[i, :,:] = clahe.apply(np.array(imgs[i,:,:], dtype = np.uint8))
   return imgs_equalized


def transform_ctdata(image, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


class para:
    # root path
    root_ct_path = '../../Data/LITS2017challenge/volume'
    root_seg_path = '../../Data/LITS2017challenge/segmentation'
    # save path
    save_path = '../data/LITS_liver_seg_processed'
    # train_index
    train_idx = [n for n in range(0, 100)]
    valid_idx = [n for n in range(100, 115)]
    test_idx = [n for n in range(115, 130)]

    size = 24
    down_scale = 0.5

    expand_slice = 20
    centercrop_size = 400

    slice_thickness = 1

    lower, upper = -200, 200  # Hu windows


class para_nnunet:
    # root path
    root_ct_path = '../../nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task03_Liver/imagesTr'
    root_seg_path = '../../nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task03_Liver/labelsTr'
    # save path
    save_path = '../data/LITS_liver_seg_org'
    # train_index
    train_idx = [n for n in range(0, 100)]
    valid_idx = [n for n in range(100, 115)]
    test_idx = [n for n in range(115, 130)]

    size = 24
    down_scale = 0.5

    expand_slice = 20
    centercrop_size = 400

    slice_thickness = 1

    lower, upper = -200, 200  # Hu windows


def processing_lits_data_downsamplex2(para, flag='train'):

    if flag == 'train':
        desired_index = para.train_idx
    elif flag == 'test':
        desired_index = para.test_idx
    else:
        desired_index = para.valid_idx

    save_dir = para.save_path + '/' + flag

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    new_ct_path = os.path.join(save_dir, 'volume')
    new_seg_dir = os.path.join(save_dir, 'segmentation')

    os.makedirs(new_ct_path)
    os.makedirs(new_seg_dir)

    ct_files = []
    all_ct_files = os.listdir(para.root_ct_path)
    assert len(all_ct_files) == 131
    for file in all_ct_files:
        index = int(file[file.find('-')+1:-4])
        if index in desired_index:
            ct_files.append(file)

    start = time()

    for file in tqdm(ct_files):

        ct = sitk.ReadImage(os.path.join(para.root_ct_path, file), sitk.sitkFloat32)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(para.root_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)


        seg_array[seg_array > 0] = 1


        ct_array[ct_array < para.lower] = para.lower
        ct_array[ct_array > para.upper] = para.upper
        ct_array = (ct_array - para.lower) / (para.upper - para.lower)

        assert np.min(ct_array) >= 0. and np.max(ct_array) <= 1.


        print('before zoom', ct_array.shape, seg_array.shape)
        print('before z', ct.GetSpacing()[-3], ct.GetSpacing()[-2], ct.GetSpacing()[-1])

        ct_array = ndimage.zoom(ct_array, (1, 0.5, 0.5), order=3)
        seg_array = ndimage.zoom(seg_array, (1, 0.5, 0.5), order=0)

        print('after zoom', ct_array.shape, seg_array.shape)
        print('after z', ct.GetSpacing()[-3], ct.GetSpacing()[-2], ct.GetSpacing()[-1])


        print('final', ct_array.shape, seg_array.shape)

        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * 2, ct.GetSpacing()[1] * 2, para.slice_thickness))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * 2, ct.GetSpacing()[1] * 2, para.slice_thickness))

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation')))


def processing_lits_data_liver_tumor_downsamplex2(para, flag='train'):

    if flag == 'train':
        desired_index = para.train_idx
    elif flag == 'test':
        desired_index = para.test_idx
    else:
        desired_index = para.valid_idx

    save_dir = para.save_path + '/' + flag

    new_ct_path = os.path.join(save_dir, 'volume')
    new_seg_dir = os.path.join(save_dir, 'segmentation_tumor')
    if not os.path.exists(new_ct_path):
        os.makedirs(new_ct_path)
    if not os.path.exists(new_seg_dir):
        os.makedirs(new_seg_dir)

    ct_files = []
    all_ct_files = os.listdir(para.root_ct_path)
    assert len(all_ct_files) == 131
    for file in all_ct_files:
        index = int(file[file.find('-')+1:-4])
        if index in desired_index:
            ct_files.append(file)

    start = time()

    for file in tqdm(ct_files):

        ct = sitk.ReadImage(os.path.join(para.root_ct_path, file), sitk.sitkFloat32)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(para.root_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)


        #seg_array[seg_array > 0] = 1

        ct_array = np.clip(ct_array, -100, 400)
        ct_array = (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))
        ct_array = ct_array * 255
        ct_array = ct_array.astype(np.uint8)
        print(ct_array.shape)

        for i in range(ct_array.shape[0]):
            ct_array[i] = cv2.equalizeHist(ct_array[i])

        ct_array = (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))

        assert np.min(ct_array) >= 0. and np.max(ct_array) <= 1.

        #对CT数据在横断面上进行降采样
        print('before zoom', ct_array.shape, seg_array.shape)
        print('before z', ct.GetSpacing()[-3], ct.GetSpacing()[-2], ct.GetSpacing()[-1])

        ct_array = ndimage.zoom(ct_array, (1, 0.5, 0.5), order=3)
        seg_array = ndimage.zoom(seg_array, (1, 0.5, 0.5), order=0)
        #
        # seg_array[seg_array <= 0.5] = 0
        # seg_array[seg_array > 1.5] = 2
        # seg_array[(seg_array > 0.5).all() and (seg_array <=1.5).all()] =1
        # print(seg_array.dtype)
        seg_array = np.round(seg_array)
        seg_array[seg_array < 0] = 0
        seg_array[seg_array > 2] = 2
        seg_array = seg_array.astype(np.uint8)


        #print('after zoom', ct_array.shape, seg_array.shape)
        #print('after z', ct.GetSpacing()[-3], ct.GetSpacing()[-2], ct.GetSpacing()[-1])


        # z = np.any(seg_array, axis=(1, 2))
        # start_slice, end_slice = np.where(z)[0][[0, -1]]
        # print(start_slice, end_slice)

        # start_slice = max(0, start_slice - para.expand_slice)
        # end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)
        #
        # ct_array = ct_array[start_slice:end_slice + 1, :, :]
        # seg_array = seg_array[start_slice:end_slice + 1, :, :]

        #print('final', ct_array.shape, seg_array.shape)

        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * 2, ct.GetSpacing()[1] * 2, para.slice_thickness))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * 2, ct.GetSpacing()[1] * 2, para.slice_thickness))

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation')))



def post_processing(pred_seg):
    pred_seg = pred_seg.astype(np.uint8)
    liver_seg = copy.deepcopy(pred_seg)
    liver_seg = measure.label(liver_seg, 4)
    props = measure.regionprops(liver_seg)

    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    liver_seg[liver_seg != max_index] = 0
    liver_seg[liver_seg == max_index] = 1

    liver_seg = liver_seg.astype(np.bool)
    morphology.remove_small_holes(liver_seg, 5e4, connectivity=2, in_place=True)
    liver_seg = liver_seg.astype(np.uint8)

    return liver_seg


def close_operator(pred_seg):
    pred_seg = pred_seg.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pred_seg = cv2.morphologyEx(pred_seg, cv2.MORPH_CLOSE, kernel)
    pred_seg = np.array(pred_seg).astype(np.uint8)
    return pred_seg


def get_msd_path_csv(root_dir, save_path, flag):
    if flag == 'train':
        idx = [n for n in range(0, 100)]
    elif flag == 'valid':
        idx = [n for n in range(100, 115)]
    else:
        idx = [n for n in range(115, 130)]

    files = glob(root_dir + '/*.npz')
    files.sort()

    flag_files = []
    for file in files:
        file_name = os.path.split(file)[1][6:-4]
        if int(file_name) in idx:
            flag_files.append([file])
            print(file_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('{}.csv'.format(os.path.join(save_path, flag)), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for file in flag_files:
            writer.writerow(file)


if __name__ == '__main__':
    params = para()

    # mask[ z, 512, 512] img [z, 256, 256]
    # processing_lits_data_downsamplex2(params, flag='train')
    #processing_lits_data_downsamplex2(params, flag='test')
    #processing_lits_data_downsamplex2(params, flag='valid')
    # processing_lits_data_liver_tumor_downsamplex2(params, flag='train')
    # processing_lits_data_liver_tumor_downsamplex2(params, flag='test')
    # processing_lits_data_liver_tumor_downsamplex2(params, flag='valid')


