import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
from trainers import *
import argparse


class byol_config:

    attr = 'class'
    gpu_ids = [0, 1]
    bentchmark = True
    manualseed = 666
    model = 'BYOL'
    network = 'unet_2d_dense'
    init_weight_type = 'kaiming'
    note = "byol_eyepacs"

    # data
    ratio = 1
    input_size = [224, 224]
    train_dataset = 'eyepacs_cl_pretask'
    eval_dataset = 'eyepacs_cl_pretask'
    im_channel = 3
    out_ch = 256
    normalization = None

    # model pre-training
    verbose = 1
    train_batch = 128
    val_batch = 32
    optimizer = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    max_queue_size = num_workers * 4
    epochs = 600
    save_model_freq = 5
    lr = 1e-4

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
    config = byol_config()
    Trainer = BYOLTrainer(config)
    Trainer.train()


