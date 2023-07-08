# coding: utf-8
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd
import csv
import random
from monai.transforms import AddChannel, Compose, RandAffine, RandRotate, RandRotate90, RandFlip, apply_transform, ToTensor

from tqdm import tqdm
#tqdm = lambda x: x

"""
For Luna classification task (NCC)
"""


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


# Some helper functions

def get_cube_from_img(img3d, center, block_size):
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size
    start_y = max(center_y - block_size / 2, 0)
    if start_y + block_size > img3d.shape[1]:
        start_y = img3d.shape[1] - block_size
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    roi_img3d = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return roi_img3d


# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def get_node_classify():
    # Getting list of image files and output nuddle 0 and 1
    for subsetindex in range(10):
        classify_size = 48
        luna_path = "../../Data/LUNA2016/"
        luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
        output_path ="../../Data/LUNA_Classification"
        file_list = glob(luna_subset_path + "*.mhd")
        print('The length of the {} th subset is {}'.format(subsetindex, len(file_list)))
        file_list_path = []
        for i in range(len(file_list)):
            file_list_path.append(file_list[i][0:-4])

        # The locations of the nodes
        # luna_csv_path = "../../Data/LUNA2016/"
        #df_node = pd.read_csv(luna_csv_path + "candidates.csv")
        #df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list_path, file_name))
        #df_node = df_node.dropna()
        #print('df node', df_node)
        df_node = {'seriesuid': [], 'coordX': [], 'coordY': [], 'coordZ': [], 'class': []}
        with open("../../Data/LUNA2016/candidates.csv")as f:
            f_csv = csv.reader(f)
            i = 0
            for row in f_csv:
                i = i + 1
                if i == 1:
                    continue
                df_node['seriesuid'].append(str(row[0]))
                df_node['coordX'].append(float(row[1]))
                df_node['coordY'].append(float(row[2]))
                df_node['coordZ'].append(float(row[3]))
                df_node['class'].append(float(row[4]))
        assert i == 551065+1
        # print(df_node['seriesuid'])
        # Looping over the image files
        for fcount, img_file in enumerate(tqdm(file_list_path)):
            # get all nodules associate with file
            mini_df = {'seriesuid':  [], 'coordX': [], 'coordY': [], 'coordZ': [], 'class': []}
            valid_file_index = [index for index, x in enumerate(df_node['seriesuid']) if x == os.path.split(img_file)[1]]
            print(valid_file_index)
            mini_df['seriesuid'] = [df_node['seriesuid'][index] for index in valid_file_index]
            mini_df['coordX'] = [df_node['coordX'][index] for index in valid_file_index]
            mini_df['coordY'] = [df_node['coordY'][index] for index in valid_file_index]
            mini_df['coordZ'] = [df_node['coordZ'][index] for index in valid_file_index]
            mini_df['class'] = [df_node['class'][index] for index in valid_file_index]
            print(mini_df)
            # some files may not have a nodule--skipping those
            if len(mini_df['seriesuid']) > 0:
                print('img_file', img_file)
                img_file = img_file + ".mhd"
                # print(img_file)
                # load the data once
                itk_img = load_itkfilewithtrucation(img_file, 600, -1000)
                img_array = sitk.GetArrayFromImage(itk_img)
                # x,y,z  Origin in world coordinates (mm)
                origin = np.array(itk_img.GetOrigin())
                # spacing of voxels in world coor. (mm)
                spacing = np.array(itk_img.GetSpacing())
                # go through all nodes
                index = 0
                for node_x, node_y, node_z, label in zip(mini_df['coordX'], mini_df['coordY'], mini_df['coordZ'], mini_df['class']):
                    # nodule center
                    center = np.array([node_x, node_y, node_z])
                    # nodule center in voxel space (still x,y,z ordering)  # clip prevents going out of bounds in Z
                    v_center = np.rint((center - origin) / spacing)
                    # convert x,y,z order v_center to z,y,x order v_center
                    v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                    # get cub size of classify_size
                    node_cube = get_cube_from_img(img_array, v_center, classify_size)
                    node_cube.astype(np.uint8)
                    # save as bmp file
                    # for i in range(classify_size):
                    #     if label == 1:
                    #         filepath = output_path + "1/" + str(subsetindex) + "_" + str(fcount) + "_" + str(index) + "/"
                    #         if not os.path.exists(filepath):
                    #             os.makedirs(filepath)
                    #         cv2.imwrite(filepath + str(i) + ".bmp", node_cube[i])
                    #     if label == 0:
                    #         filepath = output_path + "0/" + str(subsetindex) + "_" + str(fcount) + "_" + str(index) + "/"
                    #         if not os.path.exists(filepath):
                    #             os.makedirs(filepath)
                    #         cv2.imwrite(filepath + str(i) + ".bmp", node_cube[i])
                    # index += 1
                    # save as npy file
                    if label == 1:
                        print('**********lable 1**********')
                        filepath = output_path + "/1/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                        np.save(filepath + filename + ".npy", node_cube)
                    if label == 0:
                        print('*********lable 0**********')
                        filepath = output_path + "/0/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                        np.save(filepath + filename + ".npy", node_cube)
                    index += 1


def get_node_classify_org():
    # Getting list of image files and output nuddle 0 and 1
    for subsetindex in range(10):
        classify_size = 48
        luna_path = "../../Data/LUNA2016/"
        luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
        output_path = "../../Data/LUNA_Classification_org"
        file_list = glob(luna_subset_path + "*.mhd")

        file_list_path = []
        for i in range(len(file_list)):
            file_list_path.append(file_list[i][0:-4])

        # The locations of the nodes
        luna_csv_path = "../../Data/LUNA2016"
        df_node = pd.read_csv(luna_csv_path + "/candidates.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list_path, file_name))
        df_node = df_node.dropna()
        # Looping over the image files
        for fcount, img_file in enumerate(tqdm(file_list_path)):
            # get all nodules associate with file
            mini_df = df_node[df_node["file"] == img_file]
            # some files may not have a nodule--skipping those
            if mini_df.shape[0] > 0:
                img_file = img_file + ".mhd"
                print('image_file', img_file)
                # load the data once
                itk_img = load_itkfilewithtrucation(img_file, 600, -1000)
                img_array = sitk.GetArrayFromImage(itk_img)
                # x,y,z  Origin in world coordinates (mm)
                origin = np.array(itk_img.GetOrigin())
                # spacing of voxels in world coor. (mm)
                spacing = np.array(itk_img.GetSpacing())
                # go through all nodes
                index = 0
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    label = cur_row["class"]
                    print(node_x, node_y, node_z, label)
                    # nodule center
                    center = np.array([node_x, node_y, node_z])
                    # nodule center in voxel space (still x,y,z ordering)  # clip prevents going out of bounds in Z
                    v_center = np.rint((center - origin) / spacing)
                    # convert x,y,z order v_center to z,y,z order v_center
                    v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                    # get cub size of classify_size
                    node_cube = get_cube_from_img(img_array, v_center, classify_size)
                    node_cube.astype(np.uint8)
                    # save as bmp file
                    # for i in range(classify_size):
                    #     if label == 1:
                    #         filepath = output_path + "1/" + str(subsetindex) + "_" + str(fcount) + "_" + str(index) + "/"
                    #         if not os.path.exists(filepath):
                    #             os.makedirs(filepath)
                    #         cv2.imwrite(filepath + str(i) + ".bmp", node_cube[i])
                    #     if label == 0:
                    #         filepath = output_path + "0/" + str(subsetindex) + "_" + str(fcount) + "_" + str(index) + "/"
                    #         if not os.path.exists(filepath):
                    #             os.makedirs(filepath)
                    #         cv2.imwrite(filepath + str(i) + ".bmp", node_cube[i])
                    # index += 1
                    # save as npy file
                    if label == 1:
                        filepath = output_path + "1/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                        np.save(filepath + filename + ".npy", node_cube)
                    if label == 0:
                        filepath = output_path + "0/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                        np.save(filepath + filename + ".npy", node_cube)
                    index += 1



def augment(filepathX, aug_path, aug_number):
    dataX = glob(filepathX + "*.npy")
    print(len(dataX))
    if not os.path.exists(aug_path):
        os.makedirs(aug_path)

    for index in tqdm(range(len(dataX))):
        images_path = dataX[index]
        imagesample = np.load(images_path)
        # [z, y, x] [ 0, 255]
        assert  np.min(imagesample) >=0 and np.max(imagesample) <= 255 and len(imagesample.shape) == 3
        file_name = os.path.split(images_path)[1]
        train_transforms = Compose(
            [AddChannel(),
             RandAffine(prob=0.5, translate_range=(1, 1, 1),
                        padding_mode="border",
                        as_tensor_output=False),
             RandFlip(prob=0.5, spatial_axis=(1, 2)),
             RandRotate(range_x=0, range_y=20, range_z=20, prob=0.6),
             RandRotate90(prob=0.5, spatial_axes=(1, 2))])

        for num in tqdm(range(aug_number)):
            data_trans = apply_transform(train_transforms, imagesample.astype(np.float))

            data_trans = data_trans.squeeze(0).astype(np.uint8)
            print('trans', data_trans.shape, np.min(data_trans), np.max(data_trans))
            npy_path = os.path.join(aug_path, file_name[:-4] + '_aug_' + str(num) + ".npy")
            np.save(npy_path, data_trans)


def get_train_csv(root_dir, train_fold, save_path, aug_range):

    # for class 0
    class_0_dir = os.path.join(root_dir, '0')
    class_0_files = glob(class_0_dir + '/*.npy')

    class_0_train_files = []
    for class_0_file in class_0_files:
        file_name = os.path.split(class_0_file)[1][:-4]
        if int(file_name[0]) in train_fold:
            class_0_train_files.append(class_0_file)

    # random select 20%
    #total_len = len(class_0_train_files)
    #random.shuffle(class_0_train_files)
    #class_0_train_files = class_0_train_files[0:int(0.2*total_len)+1]

    print('0', class_0_train_files)
    # class 1
    class_1_dir = os.path.join(root_dir, '1')
    class_1_aug_dir = os.path.join(root_dir, '1_aug_monai')
    class_1_files = glob(class_1_dir + '/*.npy')

    class_1_train_files = []
    for class_1_file in class_1_files:
        #print(class_1_file)
        file_name = os.path.split(class_1_file)[1][:-4]
        if int(file_name[0]) in train_fold:
            class_1_train_files.append(class_1_file)
            for j in range(aug_range[0], aug_range[1]):
                aug_file = os.path.join(class_1_aug_dir, file_name+'_aug_'+str(j)+'.npy')
                #print(aug_file)
                class_1_train_files.append(aug_file)
                assert os.path.exists(aug_file)

    print('1', len(class_1_train_files))

    # save train.csv
    train_files = class_1_train_files + class_0_train_files
    train_labels = [1] * len(class_1_train_files) + [0] * len(class_0_train_files)
    assert len(train_files) == len(train_labels)
    print('total', len(train_files))

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # with open('{}.csv'.format(os.path.join(save_path, "train")), 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f, dialect='excel')
    #     for label, file in zip(train_labels, train_files):
    #         #print(label, file)
    #         writer.writerow([label, file])
    # data_frame = pd.DataFrame(data={'label': train_labels,
    #                                 'image': train_files}, index=None)
    # data_frame.to_csv(os.path.join(save_path, "train.csv"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('{}.csv'.format(os.path.join(save_path, "train_0")), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for label, file in zip([0] * len(class_0_train_files), class_0_train_files):
            # print(label, file)
            writer.writerow([label, file])

    with open('{}.csv'.format(os.path.join(save_path, "train_1")), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for label, file in zip([1] * len(class_1_train_files), class_1_train_files):
            # print(label, file)
            writer.writerow([label, file])


def get_test_csv(root_dir, test_fold, save_path, split='test'):

    # for class 0
    class_0_dir = os.path.join(root_dir, '0')
    class_0_files = glob(class_0_dir + '/*.npy')

    class_0_test_files = []
    for class_0_file in class_0_files:
        file_name = os.path.split(class_0_file)[1][:-4]
        if int(file_name[0]) in test_fold:
            class_0_test_files.append(class_0_file)

    print('0', class_0_test_files)
    # class 1
    class_1_dir = os.path.join(root_dir, '1')
    class_1_files = glob(class_1_dir + '/*.npy')

    class_1_test_files = []
    for class_1_file in class_1_files:
        #print(class_1_file)
        file_name = os.path.split(class_1_file)[1][:-4]
        if int(file_name[0]) in test_fold:
            class_1_test_files.append(class_1_file)

    print('1', len(class_1_test_files))

    # save test.csv
    test_files = class_1_test_files + class_0_test_files
    test_labels = [1] * len(class_1_test_files) + [0] * len(class_0_test_files)
    assert len(test_files) == len(test_labels)
    print('total', len(test_files))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('{}.csv'.format(os.path.join(save_path, split)), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for label, file in zip(test_labels, test_files):
            #print(label, file)
            writer.writerow([label, file])
    # data_frame = pd.DataFrame(data={'label': train_labels,
    #                                 'image': train_files}, index=None)
    # data_frame.to_csv(os.path.join(save_path, "train.csv"))


if __name__ == '__main__':

    ### Step1. First crop [48, 48, 48] cubes according to the candidates.csv
    get_node_classify()

    ### After Step1,  we can get two-category dirs: luna_classification/1 (1351) and luna_classification/0 (549714)

    ### Step2. Since serious class-imbalance exists in this task, the class 1 is augmented 40 times..

    # from pre_processing.dataaugmentation.ImageAugmentation import DataAug3D
    aug = DataAug3D(rotation=45, width_shift=0.05, height_shift=0.05, depth_shift=0, zoom_range=0)
    aug.DataAugmentation_from_dirpath('../../Data/LUNA_Classification/1/', number=40, aug_path='../../Data/LUNA_Classification/1_aug')
    ## or use monai
    # augment(filepathX='../../Data/LUNA_Classification/1/', aug_number=10, aug_path='../../Data/LUNA_Classification/1_aug_monai')

    ### After Step 2, we get luna_classification/1_aug(54040)

    ### Step3. Split training set, validation set and test set:
    ### train_fold = [0, 1, 2, 3, 4] (5*89=445)
    ### valid_fold = [5, 6](2*89=178)
    ### test_fold = [7, 8, 9](89+88*2=265)
    ### Get training dataset.
    get_train_csv(root_dir='../../Data/LUNA_Classification', train_fold=[0, 1, 2, 3, 4],
                  save_path='../../Data/LUNA_Classification/MG_split', aug_range=[0, 10])
    get_test_csv(root_dir='../../Data/LUNA_Classification', test_fold=[5, 6],
                 save_path='../../Data/LUNA_Classification/MG_split', split='valid')
    get_test_csv(root_dir='../../Data/LUNA_Classification', test_fold=[7, 8, 9],
                save_path='../../Data/LUNA_Classification/MG_split', split='test')

    #### After Step 3, we get :
    #### train.csv (0:66295 ; 1:34153; total: 100448)
