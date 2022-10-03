import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class ncc_config:
    attr = 'class'
    gpu_ids = [0, 1]
    benchmark = False
    manualseed = 111 #666
    model = 'Simple'
    network = 'unet_3d_dense'
    init_weight_type = 'kaiming'
    note = "luna_ncc_rcb_(order_xyz)"

    # data
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000
    hu_max = 600
    # input_rows = 64
    # input_cols = 64
    # input_deps = 32
    input_size = [48, 48, 48]
    train_dataset = 'luna_ncc'
    eval_dataset = 'luna_ncc'
    im_channel = 1
    class_num = 1
    normalization = 'sigmoid' # for bce loss.
    random_sample_ratio = 1 # if none, extremely imbalanced
    sample_type = 'random'

    # model
    optimizer = 'adam'
    scheduler = None
    lr = 1e-3
    patience = 10
    verbose = 1
    train_batch = 24
    val_batch = 64
    val_epoch = 5
    num_workers = 2
    max_queue_size = num_workers * 1
    epochs = 10000
    loss = 'bce'

    # pretrain
    resume = None
    pretrained_model =  None
    transferred_part = 'encoder'#'encoder'
    ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
    ## ['module.', 'module.encoder.'] for MG/PCRL
    ## None for RCB/SSM/CL/Jigsaw/BYOL
    transferred_dismatched_keys = None #['module.', 'module.encoder.']
   

    def display(self, logger):
        logger.info("Configurations")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
                # print("\n")


if __name__ == '__main__':
    config = ncc_config()
    Trainer = ClassificationTrainer(config)
    Trainer.train()
