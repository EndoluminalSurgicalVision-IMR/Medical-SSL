import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class eyepacs_config:
    attr = 'class'
    gpu_ids = [0, 1]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_2d_dense'
    init_weight_type = 'kaiming'
    note = "from_scratch"

    # data
    input_size = [512, 512]
    train_dataset = 'eyepacs_cls_per10'
    eval_dataset = 'eyepacs_cls_per10'
    im_channel = 3
    out_ch = 1
    class_num = 5
    normalization = None  # To use bce loss.
    data_aug = None #['horizontal_flip', 'vertical_flip','rotation']
    # model
    optimizer = 'adam'
    scheduler = None
    nesterov = True
    lr = 1e-4
    patience = 20  # 40
    verbose = 1
    train_batch = 24  # 24
    val_batch = 32  # 32
    val_epoch = 5
    num_workers = 8
    max_queue_size = num_workers * 1
    epochs = 600
    nb_epochs = 300
    loss = 'mse'


    # pretrain
    resume = None
    pretrained_model = None
    transferred_part = 'encoder'
    transferred_dismatched_keys = None  # ['module.', 'module.encoder.']
    fine_tuning_scheme = 'full'

    def display(self, logger):
        logger.info("Configurations")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
                # print("\n")


if __name__ == '__main__':
    config = eyepacs_config()
    Trainer = Classification2DTrainer(config)
    Trainer.train()


