import os
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from trainers import *
import argparse


class ssm_rpl_config:
    attr = 'class'
    gpu_ids = [0]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_2d_rpl'
    init_weight_type = 'kaiming'
    note = "SSM_RPL"
    ratio = 1

    # data
    input_size = [512, 512]
    train_dataset = 'eyepacs_rpl_pretask'
    eval_dataset = 'eyepacs_rpl_pretask'
    im_channel = 3
    num_grids_per_axis = 3
    class_num = 8
    out_ch = 8
    normalization = None  # for cross_entropy loss

    # model pre-training
    verbose = 1
    train_batch = 32
    val_batch = 32
    optimizer = "adam"
    # scheduler = 'StepLR'
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [150]
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 100000
    save_model_freq = 20
    # patience = 1000
    val_epoch = 5
    lr = 0.001
    loss = 'ce'

    # logs
    resume = None
    pretrained_model = None

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = ssm_rpl_config()
    Trainer = RPLTrainer(config)
    Trainer.train()
 