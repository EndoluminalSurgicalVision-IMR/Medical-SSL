import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from trainers import *
import argparse


class pcrl_config:

    attr = 'class'
    gpu_ids = [0]
    bentchmark = True
    manualseed = 666
    model = 'PCRL_Model2d'
    network = 'pcrl_2d'
    init_weight_type = 'kaiming'
    note = "mypcrl_2d"

    # data
    ratio = 1
    input_size = [224, 224]
    train_dataset = 'eyepacs_pcrl_pretask'
    eval_dataset = 'eyepacs_pcrl_pretask'
    im_channel = 3
    class_num = 1

    # model pre-training
    verbose = 1
    train_batch = 32
    val_batch = 32
    optimizer = "sgd"
    momentum = 0.9
    # gamma = 0.5
    weight_decay = 1e-4
    nesterov = False
    scheduler = None
    num_workers = 4
    max_queue_size = num_workers * 4
    epochs = 240
    save_model_freq = 5
    patience = 50
    lr = 1e-3
    #step = 50
    #lr_decay_epochs = '160,200,240,280,320'
    #lr_decay_rate = 0.1
    loss = 'mse'
    moco_t = 0.2
    use_amp = True

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
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = pcrl_config()
    Trainer = MYPCRL3DTrainer(config)
    Trainer.train()

