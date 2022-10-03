import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from trainers import *
import argparse


class ssm_rot_config:

    attr = 'class'
    gpu_ids = [0, 1]
    bentchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_3d_dense'
    init_weight_type = 'kaiming'
    note = "SSM_Rot_random_crop_64_64_32_v2"

    # data
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_size = [64, 64, 64]#[48, 48, 48]#[128, 128, 128]# [64, 64, 64]
    org_data_size = [64, 64, 32]#[48, 48, 32]#[128, 128, 64]# [64, 64, 32]
    train_dataset = 'luna_rot_pretask_v2'
    eval_dataset = 'luna_rot_pretask_v2'
    im_channel = 1
    class_num = 10
    normalization = None # for cross_entropy loss
    num_rotations_per_patch = 1 # 4 # multiple versions for one patch

    # model pre-training
    verbose = 1
    train_batch = 2
    val_batch = 2
    val_epoch = 10
    optimizer = "adam"
    # scheduler = 'StepLR'
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [250]
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 1000
    save_model_freq = 30
    patience = 50
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
    config = ssm_rot_config()
    Trainer = RotTrainer(config)
    Trainer.train()


