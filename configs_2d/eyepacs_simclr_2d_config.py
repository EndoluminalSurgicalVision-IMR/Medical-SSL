import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from trainers import *
import argparse


class simclr_config:

    attr = 'class'
    gpu_ids = [0]
    bentchmark = True
    manualseed = 666
    model = 'SimCLR'
    network = 'unet_2d_dense'
    init_weight_type = 'kaiming'
    note = "simclr_eyepacs"

    # data
    ratio = 1
    input_size = [128, 128]
    train_dataset = 'eyepacs_cl_pretask'
    eval_dataset = 'eyepacs_cl_pretask'
    im_channel = 3
    out_ch = 256
    normalization = None

    # model pre-training
    verbose = 1
    train_batch = 256 #128
    val_batch = 16
    optimizer = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    max_queue_size = num_workers * 4
    epochs = 500
    save_model_freq = 5
    lr = 0.05

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
    config = simclr_config()
    Trainer = SimCLRTrainer(config)
    Trainer.train()




