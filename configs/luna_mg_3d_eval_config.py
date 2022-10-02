import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from testers import *
import argparse


class models_genesis_config:

    attr = 'class'
    gpu_ids = [0, 1]
    benchmark = False
    manualseed = 666
    model = 'Simple'
    network = 'unet_3d' # 'unet_3d_wo_skip'
    # use_MLP = False
    init_weight_type = 'kaiming'
    note = "genesis_chest_ct"

    # data
    #data = "../../Data/Self_Learning_Cubes_1.0/bat_32_s_64x64x32"
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    # input_rows = 64
    # input_cols = 64
    # input_deps = 32
    input_size = [64, 64, 32]
    train_dataset = 'luna_mg_pretask'
    eval_dataset = 'luna_mg_pretask'
    im_channel = 1
    class_num = 1
    normalization = 'sigmoid'
    loss = 'mse'
    val_batch = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4

    # logs
    model_path = '../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_212.pth'
    save_results_path = '../results/luna_mg_eval_images'

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")


if __name__ == '__main__':
    config = models_genesis_config()
    Trainer = MGTester(config)
    Trainer.compare_val_imgs('unet_3d_wo_skip', '../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth')


