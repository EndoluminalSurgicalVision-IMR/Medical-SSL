import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse



class ncs_config:

    attr = 'class'
    object = 'lung_nodule'

    gpu_ids = [0, 1]
    benchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_3d'
    init_weight_type = 'kaiming'
    note = "luna_ncs_from_scratch(kaiming)"

    # data
    input_size = [64, 64, 32]
    train_dataset = 'luna_ncs'
    eval_dataset = 'luna_ncs'
    im_channel = 1
    class_num = 1
    normalization = 'sigmoid'  # To use bce loss.

    # model
    optimizer = 'adam'
    scheduler = None
    lr = 1e-3
    patience = 50
    verbose = 1
    train_batch = 16
    val_batch = 16
    val_epoch = 5
    num_workers = 1
    max_queue_size = num_workers * 1
    epochs = 10000
    loss = 'dice'

    # pretrain
    resume = None
    pretrained_model = None
    transferred_part = 'all'
    ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
    transferred_dismatched_keys = None

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = ncs_config()
    Trainer = Seg3DROITrainer(config)
    Trainer.train()
