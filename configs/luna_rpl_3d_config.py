import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *
import argparse


class ssm_rpl_config:

    attr = 'class'
    gpu_ids = [0, 1]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_3d_rpl'
    init_weight_type = 'kaiming'
    note = "SSM_RPL_96"

    # data
    #data = "../../Data/Self_Learning_Cubes_1.0/bat_32_s_64x64x32"
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 4
    input_size = [96, 96, 16]#[64, 64, 16]
    org_data_size = [320, 320, 74]
    train_dataset = 'luna_rpl_pretask_v2'
    eval_dataset = 'luna_rpl_pretask_v2'
    im_channel = 1
    num_grids_per_axis = 3
    class_num = 26
    normalization = None # for cross_entropy loss

    # model pre-training
    verbose = 1
    train_batch = 16
    val_batch = 16
    optimizer = "adam"
    # scheduler = 'StepLR'
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [250]
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 100000
    save_model_freq = 50
    # patience = 1000
    val_epoch = 50
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
    config = ssm_rpl_config()
    Trainer = RPLTrainer(config)
    Trainer.train()


