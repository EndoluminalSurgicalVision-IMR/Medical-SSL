
import copy
import random

import numpy as np
from torch.utils.data import Dataset
import torchio.transforms


class PTPBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.base_dir = base_dir
        self.all_images = []
        self.flag = flag
        self.crop_size = self.config.input_size

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

    def random_crop(self, image, crop_size):
        """Crop a patch from an image randomly.
                    Args:
                        image: [C, D, H, W]
                        crop_size: the desired output size [d, h, w]
                  Returns:
                      out_image:[C, d, h, w]
        """
        _, h, w, d = image.shape
        if np.random.uniform() > 0.33:
            h1 = np.random.randint((h - crop_size[0]) // 4, 3 * (h - crop_size[0]) // 4)
            w1 = np.random.randint((w - crop_size[1]) // 4, 3 * (w - crop_size[1]) // 4)

        else:
            h1 = np.random.randint(0, h - crop_size[0])
            w1 = np.random.randint(0, w - crop_size[1])

        d1 = np.random.randint(0, d - crop_size[2])

        image = image[:, h1:h1 + crop_size[0], w1:w1 + crop_size[1], d1:d1 + crop_size[2]]

        return image

    def center_crop(self, image, label, size):
        """CenterCrop an image.
           Args:
              image: [C, H, W, D]
              label:[C, H, W, D]
              crop_size: the desired output size [d, h, w]
           Returns:
              out_image:[C, d, h, w]
              out_label:[K, d, h, D]
        """
        _, h, w, d = image.shape

        w1 = int(round((w - size) / 2.))
        h1 = int(round((h - size) / 2.))

        image = image[:, h1:h1 + size, w1:w1 + size, :]
        label = label[:, h1:h1 + size, w1:w1 + size, :]
        return image, label

    def crop_3d(self, image, flag, crop_size):
        h, w, d = crop_size[0], crop_size[1], crop_size[2]
        h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]

        if flag == 'train':
            # crop random
            x = np.random.randint(0, 1 + h_old - h)
            y = np.random.randint(0, 1 + w_old - w)
            z = np.random.randint(0, 1 + d_old - d)
        else:
            # crop center
            x = int((h_old - h) / 2)
            y = int((w_old - w) / 2)
            z = int((d_old - d) / 2)

        return self.do_crop_3d(image, x, y, z, h, w, d)

    def do_crop_3d(self, image, x, y, z, h, w, d):
        assert type(x) == int, x
        assert type(y) == int, y
        assert type(z) == int, z
        assert type(h) == int, h
        assert type(w) == int, w
        assert type(d) == int, d

        return image[x:x + h, y:y + w, z:z + d]

    def crop_patches_3d(self, image, patches_per_axis, patch_jitter_xy=3, patch_jitter_z=3):
        h, w, d = image.shape

        patch_overlap = -patch_jitter_xy if patch_jitter_xy < 0 else 0

        h_grid = (h - patch_overlap) // patches_per_axis
        w_grid = (w - patch_overlap) // patches_per_axis
        d_grid = (d - patch_overlap) // patches_per_axis
        # h_patch = h_grid - patch_jitter_xy
        # w_patch = w_grid - patch_jitter_xy
        # d_patch = d_grid - patch_jitter_z

        h_patch = self.crop_size[0]
        w_patch = self.crop_size[1]
        d_patch = self.crop_size[2]

        cubes = []
        for i in range(patches_per_axis):
            for j in range(patches_per_axis):
                for k in range(patches_per_axis):

                    p = self.do_crop_3d(image,
                                   i * h_grid,
                                   j * w_grid,
                                   k * d_grid,
                                   h_grid + patch_overlap,
                                   w_grid + patch_overlap,
                                   d_grid + patch_overlap)

                    if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                        p = self.crop_3d(p, self.flag, [h_patch, w_patch, d_patch])

                    cubes.append(p)

        return cubes

    def rotate_3d(self, img, rot_class):
        # 10 classes: [x, y, z] [0:3, 0:3, 0:3]
        if rot_class == 0:  # 0 degrees rotation
            return img
        elif rot_class in [1, 2, 3]:
            # along z
            return np.rot90(img, k=rot_class, axes=(-3, -2)).copy()
        elif rot_class in [4, 5, 6]:
            # along y
            return np.rot90(img, k=rot_class-3, axes=(-3, -1)).copy()
        elif rot_class in [7, 8, 9]:
            # along x
            return np.rot90(img, k=rot_class-6, axes=(-2, -1)).copy()
        else:
            raise ValueError('rot class index should be in the range [0, 10)')

    def rotate_3d_v2(self, img):
        rot = np.random.random_integers(10) - 1
        volume = copy.deepcopy(img)
        # volume: [C, H, W, D]
        if rot == 0:
            volume = volume
        elif rot == 1:
            volume = np.transpose(np.flip(volume, 2), (0, 2, 1, 3))  # 90 deg Z
        elif rot == 2:
            volume = np.flip(volume, (1, 2))  # 180 degrees on z axis
        elif rot == 3:
            volume = np.flip(np.transpose(volume, (0, 2, 1, 3)), 2)  # 90 deg Z
        elif rot == 4:
            volume = np.transpose(np.flip(volume, 2), (0, 1, 3, 2))  # 90 deg X
        elif rot == 5:
            volume = np.flip(volume, (2, 3))  # 180 degrees on x axis
        elif rot == 6:
            volume = np.flip(np.transpose(volume, (0, 1, 3, 2)), 2)  # 90 deg X
        elif rot == 7:
            volume = np.transpose(np.flip(volume, 1), (0, 3, 2, 1))  # 90 deg Y
        elif rot == 8:
            volume = np.flip(volume, (1, 3))  # 180 degrees on y axis
        elif rot == 9:
            volume = np.flip(np.transpose(volume, (0, 3, 2, 1)), 1)  # 90 deg Y
        return volume, rot




