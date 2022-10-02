import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from trainers import *


class Auto_Encoder_config:

    attr = 'class'
    gpu_ids = [0]
    benchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_3d'
    init_weight_type = 'kaiming'
    note = "auto_encoder"

    # data
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_size = [64, 64, 32]
    train_dataset = 'luna_ae_pretask'
    eval_dataset = 'luna_ae_pretask'
    im_channel = 1
    class_num = 1
    normalization = 'sigmoid'

    # model pre-training
    verbose = 1
    train_batch = 6
    val_batch = 6
    optimizer = "sgd"
    scheduler = 'StepLR'
    momentum = 0.9
    weight_decay = 0.0
    # nesterov = False
    num_workers = 10
    max_queue_size = num_workers * 4
    epochs = 10000
    # save_model_freq = 500
    patience = 50
    lr = 1
    loss = 'mse'

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
    config = Auto_Encoder_config()
    Trainer = MGTrainer(config)
    Trainer.train()


