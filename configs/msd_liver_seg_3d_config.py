import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
from trainers import *
import argparse

class msdliver_config:
    attr = 'class'
    object = 'liver_tumor'
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
    order = 'zyx' 
    note = "msd_liver_seg"

    # data [c, d, h, w] input_size: [d, h, w]
    input_size = [32, 160, 160]
    test_cut_params = {'patch_d': 32,
                       'patch_h': 160,
                       'patch_w': 160,
                       'stride_d': 16,
                       'stride_h': 80,
                       'stride_w': 80}

    train_dataset = 'msd_liver_seg_train_down2'
    eval_dataset = 'msd_liver_seg_test_down2'
    im_channel = 1
    class_num = 3
    normalization = None # To use bce loss.

    # model
    optimizer = 'adam'
    lr = 1e-2
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [150, 350] 
    patience = 20
    verbose = 1
    train_batch = 4
    train_patch_fg_ratio = 0.8
    val_batch = 1
    val_freq = 5
    save_model_freq = 50
    num_workers = 1
    max_queue_size = num_workers * 1
    epochs = 10000
    bce_weight = 1
    loss = 'nnunet_ce_dice2'
    save_val_results = True
    # pretrain
    resume = None
    pretrained_model = None
    transferred_part = None #'encoder'
    ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
    transferred_dismatched_keys = None #['module.encoder.', 'module.']

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")

if __name__ == '__main__':
    config = msdliver_config()
    Trainer = Seg3DMCTrainer(config)
    Trainer.train()






