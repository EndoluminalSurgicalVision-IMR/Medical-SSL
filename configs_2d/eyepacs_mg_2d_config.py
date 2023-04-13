import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from trainers import *
import argparse


class models_genesis_config:

    attr = 'class'
    gpu_ids = [0]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_2d' 
    init_weight_type = 'kaiming'
    note = "auto_encoder"

    # data
    input_size = [512, 512]
    train_dataset = 'eyepacs_mg_pretask'
    eval_dataset = 'eyepacs_mg_pretask'
    im_channel = 3
    class_num = 3
    out_ch = 3
    normalization = 'sigmoid'
    ratio = 1

    # model pre-training
    verbose = 1
    train_batch = 8 #6
    val_batch = 16
    optimizer = "sgd"
    scheduler = 'StepLR'
    momentum = 0.9
    weight_decay = 0.0
    nesterov = False
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 10000
    save_model_freq = 10
    patience = 50
    lr = 1
    loss = 'mse'

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4

    # logs
    resume = None
     pretrained_model = None

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = models_genesis_config()
    Trainer = MGTrainer(config)
    Trainer.train()


