import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class jigsaw_config:

    attr = 'class'
    gpu_ids = [0, 1]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_2d_jigsaw'
    init_weight_type = 'kaiming'
    note = "JigSaw_same_scale"
    ratio = 1

    # data
    input_size = [360, 360] 
    train_dataset = 'eyepacs_jigsaw_pretask'
    eval_dataset = 'eyepacs_jigsaw_pretask'
    im_channel = 3
    order_class_num = 100
    out_ch = 100
    k_permutations_path = "../datasets_2D/Jigsaw/permutations_hamming_max_100.npy"
    gaps = [6, 6]
    num_grids_per_axis = 3

    # model pre-training
    train_batch = 48#32
    val_batch = 48#32
    optimizer = "adam"
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [200]
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 1000
    save_model_freq = 50
    patience = 20
    lr = 1e-3
    loss = 'ce'
    normalization = None

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
    config = jigsaw_config()
    Trainer = JigSawTrainer(config)
    Trainer.train()






 