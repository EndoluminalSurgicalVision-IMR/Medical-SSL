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
    patience = 10#10
    verbose = 1
    train_batch = 24
    val_batch = 64
    val_epoch = 5
    num_workers = 2
    max_queue_size = num_workers * 1
    epochs = 10000
    loss = 'bce'

    # pretrain
    resume = '../checkpoints/luna_ncc/unet_3d_dense_Simple_luna_ncc_rcb_(order_xyz)_repeat/20220812-223552/model_best.pth'#'../checkpoints/luna_ncc/unet_3d_dense_Simple_luna_ncc_SSM_ROT_48(order_xyz)_V2_repeat/20220804-110123/model_best.pth'#'../checkpoints/luna_ncc/unet_3d_dense_Simple_luna_ncc_RKB_100_orders_128(order_xyz)/20220721-151436/model_best.pth'#'../checkpoints/luna_ncc/unet_3d_dense_Simple_luna_ncc_BYOL(order_xyz)/20220620-105729/model_best.pth'
    pretrained_model =  None#'../checkpoints/luna_rkb_pretask/unet_3d_rkb_Simple_RKB_128_128_32_ct/20220720-233027/RKB_100_order_CT.pth'#'../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220618-203019/470.pth'#'../checkpoints/luna_ssm_rot_pretask_v2/unet_3d_dense_Simple_SSM_Rot_random_crop_48_48_48_v2/SSM_ROT_v2_48_CT.pth'#'../checkpoints/luna_ssm_rpl_pretask/SSM_RPL48_CT.pth'#'../checkpoints/luna_rkbp_pretask/unet_3d_rkbp_Simple_RKBPlus_128_128_32_ct/20220723-202040/RKBP_300_CT.pth' #'../checkpoints/luna_rkb_pretask/unet_3d_rkb_Simple_RKB_128_128_32_ct/20220720-233027/RCB_CT.pth' #'../checkpoints/luna_jigsaw_pretask/unet_3d_jigsaw_Simple_JigSaw_resize_half_128_128_32_ct/20220714-082932/RCB_CT.pth' #'../checkpoints/luna_cl_pretask/unet_3d_dense_SimCLR_simclr_luna/20220621-101217/380.pth'#'../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220618-203019/470.pth'#'../checkpoints/luna_ssm_rpl_pretask/SSM_RPL64_CT.pth' #'../checkpoints/luna_pcrl_pretask/pcrl_3d_PCRL_Model_mypcrl_luna_random_aug_tensor/20220531-221542/95.pth' #'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_rcb_pretask/unet_3d_rcb_Simple_RCB_ct/20220418-083822/RCB_CT_473.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_pcrl_pretask/mypcrl_100.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'
    transferred_part = 'encoder'#'encoder'
    ## pretrained model keys: transferred_dismatched_keys[0A]; fine-tuned model keys: transferred_dismatched_keys[1]
#   ## ['module.', 'module.encoder.'] for mg/pcrl
    ## None for RCB/SSM/CL/Jigsaw/BYOL
    transferred_dismatched_keys = None#['module.', 'module.encoder.']
    # transfer_bn_buffer = True
    # transfer_bn_all = True

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
