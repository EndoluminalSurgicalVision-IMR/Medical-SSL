import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class byol_config:

    attr = 'class'
    gpu_ids = [0, 1]
    benchmark = True
    manualseed = 666
    model = 'BYOL'
    network = 'unet_3d_dense'
    init_weight_type = 'kaiming'
    note = "byol_luna"

    # data
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 16
    ratio = 1.0
    input_size = [64, 64, 32]
    train_dataset = 'luna_cl_pretask'
    eval_dataset = 'luna_cl_pretask'
    im_channel = 1
    class_num = 256
    normalization = None

    # model pre-training
    verbose = 1
    train_batch = 64
    val_batch = 16
    optimizer = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    max_queue_size = num_workers * 4
    epochs = 600
    save_model_freq = 5
    lr = 1e-4#0.05

    # logs
    resume = '../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220618-143852/300.pth'#'../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220617-110048/290.pth'#'../checkpoints/luna_pcrl_pretask/pcrl_3d_PCRL_Model_mypcrl_luna/20220328-235653/10.pth'
    pretrained_model = None

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = byol_config()
    Trainer = BYOLTrainer(config)
    Trainer.train()

