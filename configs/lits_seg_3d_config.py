import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse

class lcs_config:
    attr = 'class'
    object = 'liver'
    train_idx = [n for n in range(0, 100)]
    valid_idx = [n for n in range(100, 115)]
    test_idx = [n for n in range(115, 130)]
    num_train = len(train_idx)
    num_valid = len(valid_idx)
    num_test = len(test_idx)
    gpu_ids = [0, 1]
    benchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_3d'
    init_weight_type = 'kaiming'
    order = 'zyx' #'yzx'
    note = "lits_seg_mg_pretrained"

    # data [c, d, h, w] input_size: [d, h, w]
    input_size = [32, 256, 256]
    # test_cut_params = {'patch_d': 256,
    #                    'patch_h': 256,
    #                    'patch_w': 32,
    #                    'stride_d': 128,
    #                    'stride_h': 128,
    #                    'stride_w': 16}
    test_cut_params = {'patch_d': 32,
                       'patch_h': 256,
                       'patch_w': 256,
                       'stride_d': 16,
                       'stride_h': 256,
                       'stride_w': 256}

    train_dataset = 'lits_seg_train'
    eval_dataset = 'lits_seg_test'
    im_channel = 1
    class_num = 1
    normalization = 'sigmoid'  # To use bce loss.

    # model
    optimizer = 'adam'
    lr = 1e-2
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [50, 150] #only for wo skip#[50, 150]
    patience = 40#20
    verbose = 1
    train_batch = 4
    val_batch = 1
    val_freq = 5
    save_model_freq = 50
    num_workers = 1
    max_queue_size = num_workers * 1
    epochs = 10000
    loss = 'bcedice'
    dice_weight = 1
    # pretrain
    resume = None 
    pretrained_model = None
    transferred_part = 'encoder'
    ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
    transferred_dismatched_keys = ['module.encoder.', 'module.']

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = lcs_config()
    Trainer = Seg3DTrainer(config)
    Trainer.train()
