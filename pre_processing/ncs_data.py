import os
import random
import copy
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage.transform import resize
from utils.tools import save_np2nii


def load_image(data, status=None):
    x = np.squeeze(np.load(os.path.join(data, 'x_' + status + '_64x64x32.npy')))
    y = np.squeeze(np.load(os.path.join(data, 'm_' + status + '_64x64x32.npy')))
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    return x, y


if __name__ == "__main__":

    x_valid, y_valid = load_image(data='../data/ncs_data', status='train')
    print(x_valid.shape, np.min(x_valid), np.max(x_valid))
    print(y_valid.shape, np.min(y_valid), np.max(y_valid))
    for i in range(3000, 3040):
        save_np2nii(savedImg=x_valid[i].squeeze(), saved_name='image'+str(i), saved_path='../results/Luna_cls_train')
        save_np2nii(savedImg=np.array(y_valid[i].squeeze(), dtype=np.uint8), saved_name='mask'+str(i), saved_path='../results/Luna_cls_train')

        assert np.max(y_valid[i]) == 1

