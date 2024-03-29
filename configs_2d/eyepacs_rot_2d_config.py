import os
import numpy as np
from tqdm import tqdm
from trainers import *
import argparse


class ssm_rot_config:
    attr = 'class'
    gpu_ids = [0]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_2d_dense'
    init_weight_type = 'kaiming'
    note = "SSM_Rot"
    ratio =1

    # data
    input_size = [512, 512]  
    train_dataset = 'eyepacs_rot_pretask'
    eval_dataset = 'eyepacs_rot_pretask'
    im_channel = 3
    out_ch = 4
    class_num = 4
    normalization = None  # for cross_entropy loss

    # model pre-training
    verbose = 1
    train_batch = 16
    val_batch = 16
    val_epoch = 10
    optimizer = "adam"
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [250]
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 1000
    save_model_freq = 30
    patience = 50
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
    config = ssm_rot_config()
    Trainer = RotTrainer(config)
    Trainer.train()


 