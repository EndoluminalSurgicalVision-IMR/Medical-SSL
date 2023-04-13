import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
from trainers import *
import argparse


class drive_config:
    attr = 'class'
    gpu_ids = [0, 1]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_2d'
    init_weight_type = 'kaiming'
    note = "from_scratch"

    # data
    train_dataset = 'drive_seg'
    eval_dataset = 'drive_seg'
    im_channel = 3
    out_ch = 1
    class_num = 1
    normalization = 'sigmoid'  # To use bce loss.
    # model
    optimizer = 'adam'
    scheduler = None
    nesterov = True
    lr = 1e-4
    patience = 40
    verbose = 1
    train_batch = 2
    val_batch = 2
    val_epoch = 5
    num_workers = 8
    max_queue_size = num_workers * 1
    epochs = 600
    nb_epochs = 300
    loss = 'dice'
    # pretrain
    resume = None
    pretrained_model = None
    transferred_part = None
    transferred_dismatched_keys = None
    fine_tuning_scheme = 'full'

    def display(self, logger):
        logger.info("Configurations")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))

if __name__ == '__main__':
    config = drive_config()
    Trainer = Seg2DROITrainer(config)
    Trainer.train()
