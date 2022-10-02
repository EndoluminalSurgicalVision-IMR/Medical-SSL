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
    resume = '../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_mg_pretrained/20220427-084926/best_model.pth'#'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_byol_online_pretrained/20220625-160544/best_model.pth'#'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_rcb_om_pretrained/20220512-011318/best_model.pth'#'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_rcb_pretrained/20220429-021523/best_model.pth'#'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_mg_pretrained_wo_skip/20220425-213101/best_model.pth'#'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_mg_pretrained_only_encoder/20220420-161317/best_model.pth'
    #'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_from_scratch(kaiming)/20220325-163301/best_model.pth' #'../checkpoints/lits_seg_train/unet_3d_Simple_lits_seg_from_scratch(kaiming)/20220319-205127/best_model.pth'
    pretrained_model = None#'../checkpoints/luna_rkb_pretask/unet_3d_rkb_Simple_RKB_128_128_32_ct/20220720-233027/RKB_100_order_CT.pth'#'../checkpoints/luna_ncc/unet_3d_dense_Simple_luna_ncc_RKBP_100_orders_128(order_xyz)/20220724-225048/model_best.pth' #'../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220618-203019/400.pth'#'../checkpoints/luna_cl_pretask/unet_3d_dense_SimCLR_simclr_luna/20220621-101217/380.pth'#'../checkpoints/luna_ssm_rpl_pretask/SSM_RPL96_CT.pth'#'../checkpoints/luna_pcrl_pretask/mypcrl_wo_skip_100.pth'#'../checkpoints/luna_rcb_om_pretask/unet_3d_rcb_om_Simple_RCB_OM_ct/20220510-203611/RCB_CT_532.pth' #'../checkpoints/luna_rcb_pretask/unet_3d_rcb_Simple_RCB_ct/20220418-083822/RCB_CT_473.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth'#'../checkpoints/luna_pcrl_pretask/mypcrl_100.pth'#'../checkpoints/luna_pcrl_pretask/pcrl_3d_PCRL_Model_mypcrl_luna/20220329-162814/100.pth' #'../checkpoints/luna_pcrl_pretask/pcrl_3d_EMA_Model_pcrl_luna/20220322-190451/65.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'
    transferred_part = 'encoder'
    # transfer_bn_all = False
    # transfer_bn_buffer = True
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