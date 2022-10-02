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
    order = 'zyx' #'yzx'
    note = "msd_liver_seg_RKB_orders_100_nnunet_stage0"

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
    normalization = None #'softmax'  # To use bce loss.

    # model
    optimizer = 'adam'
    lr = 1e-2
    scheduler = 'StepLR_multi_step'
    learning_rate_decay = [150, 350] # [150, 300] only for wo skip
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
    resume = '../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_RKB_orders_100_nnunet_stage0/20220729-114219/best_model.pth' #'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_JigSaw_128_nnunet_stage0/20220716-190356/best_model.pth'# '../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_byol_online_nnunet_stage0/20220710-091646/latest_model.pth' # '../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_byol_online_nnunet_stage0/20220709-193254/latest_model.pth' # '../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_byol_nnunet_stage0/20220620-112517/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_rpl_96_nnunet_stage0/20220612-200223/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_rpl_48_nnunet_stage0/20220611-231506/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_rpl_nnunet_stage0/20220610-142522/latest_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_pcrl_nnunet_stage0/20220609-112205/best_model.pth' #'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220605-212515/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_rcb_nnunet_stage0/20220608-211643/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_ssm_rot128_nnunet_stage0/20220608-211612/best_model.pth'
    pretrained_model = None#'../checkpoints/luna_ssm_rpl_pretask/SSM_RPL96_CT.pth'#'../checkpoints/luna_rkb_pretask/unet_3d_rkb_Simple_RKB_128_128_32_ct/20220720-233027/RKB_100_order_CT.pth'#'../checkpoints/luna_jigsaw_pretask/unet_3d_jigsaw_Simple_JigSaw_resize_half_128_128_32_ct/20220714-082932/RCB_CT.pth'# '../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220618-203019/400.pth'#'../checkpoints/luna_cl_pretask/unet_3d_dense_BYOL_byol_luna/20220618-203019/470.pth'#'../checkpoints/luna_ssm_rpl_pretask/SSM_RPL96_CT.pth' # '../checkpoints/luna_rcb_pretask/unet_3d_rcb_Simple_RCB_ct/20220418-083822/RCB_CT_473.pth'#'../checkpoints/luna_rcb_om_pretask/unet_3d_rcb_om_Simple_RCB_OM_ct/20220510-203611/RCB_CT_532.pth' #'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth'#'../checkpoints/luna_pcrl_pretask/mypcrl_100.pth'#'../checkpoints/luna_pcrl_pretask/pcrl_3d_PCRL_Model_mypcrl_luna/20220329-162814/100.pth' #'../checkpoints/luna_pcrl_pretask/pcrl_3d_EMA_Model_pcrl_luna/20220322-190451/65.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'
    transferred_part = None#'encoder'
    ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
    transferred_dismatched_keys = None#['module.encoder.', 'module.']

    def display(self, logger):
        """Display Configuration values."""
        logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                logger.info("{:30} {}".format(a, getattr(self, a)))
        logger.info("\n")

#
# class msdliver_config2:
#     attr = 'class'
#     object = 'liver_tumor'
#     train_idx = [n for n in range(0, 100)]
#     valid_idx = [n for n in range(100, 115)]
#     test_idx = [n for n in range(115, 130)]
#     num_train = len(train_idx)
#     num_valid = len(valid_idx)
#     num_test = len(test_idx)
#     gpu_ids = [0, 1]
#     bentchmark = False
#     manualseed = 666
#     model = 'Simple'
#     network = 'unet_3d'
#     init_weight_type = 'kaiming'
#     order = 'zyx' #'yzx'
#     note = "msd_liver_seg_fs_nnunet_stage0_v2"
#
#     # data [c, d, h, w] input_size: [d, h, w]
#     input_size = [32, 160, 160]
#     test_cut_params = {'patch_d': 32,
#                        'patch_h': 160,
#                        'patch_w': 160,
#                        'stride_d': 16,
#                        'stride_h': 80,
#                        'stride_w': 80}
#
#     train_dataset = 'msd_liver_seg_train_down2'
#     eval_dataset = 'msd_liver_seg_test_down2'
#     im_channel = 1
#     class_num = 3
#     normalization = None #'softmax'  # To use bce loss.
#
#     # model
#     optimizer = 'adam'
#     lr = 1e-2
#     scheduler = 'StepLR_multi_step'
#     learning_rate_decay = [150, 350] # [150, 300] only for wo skip
#     patience = 20
#     verbose = 1
#     train_batch = 4
#     train_patch_fg_ratio = 0.8
#     val_batch = 1
#     val_freq = 5
#     save_model_freq = 50
#     num_workers = 1
#     max_queue_size = num_workers * 1
#     epochs = 10000
#     dice_weight = [0.5, 1, 10]
#     bce_weight = 1
#     loss = 'nnunet_ce_dice2'
#     save_val_results = True
#     # pretrain
#     resume = None#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220531-123611/latest_model.pth' # '../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220531-001039/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220530-155939/best_model.pth'
#     pretrained_model = None #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_rcb_om_pretask/unet_3d_rcb_om_Simple_RCB_OM_ct/20220510-203611/RCB_CT_532.pth' #'../checkpoints/luna_rcb_pretask/unet_3d_rcb_Simple_RCB_ct/20220418-083822/RCB_CT_473.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth'#'../checkpoints/luna_pcrl_pretask/mypcrl_100.pth'#'../checkpoints/luna_pcrl_pretask/pcrl_3d_PCRL_Model_mypcrl_luna/20220329-162814/100.pth' #'../checkpoints/luna_pcrl_pretask/pcrl_3d_EMA_Model_pcrl_luna/20220322-190451/65.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'
#     transferred_part = 'all'
#     ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
#     transferred_dismatched_keys = None#['module.encoder.', 'module.']
#
#     def display(self, logger):
#         """Display Configuration values."""
#         logger.info("\nConfigurations:")
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
#                 logger.info("{:30} {}".format(a, getattr(self, a)))
#         logger.info("\n")
#
#
# class msdliver_config3:
#     attr = 'class'
#     object = 'liver_tumor'
#     train_idx = [n for n in range(0, 100)]
#     valid_idx = [n for n in range(100, 115)]
#     test_idx = [n for n in range(115, 130)]
#     num_train = len(train_idx)
#     num_valid = len(valid_idx)
#     num_test = len(test_idx)
#     gpu_ids = [0, 1]
#     bentchmark = False
#     manualseed = 666
#     model = 'Simple'
#     network = 'unet_3d'
#     init_weight_type = 'kaiming'
#     order = 'zyx' #'yzx'
#     note = "msd_liver_seg_fs_nnunet_stage0_v3"
#
#     # data [c, d, h, w] input_size: [d, h, w]
#     input_size = [32, 160, 160]
#     test_cut_params = {'patch_d': 32,
#                        'patch_h': 160,
#                        'patch_w': 160,
#                        'stride_d': 16,
#                        'stride_h': 80,
#                        'stride_w': 80}
#
#     train_dataset = 'msd_liver_seg_train_down2'
#     eval_dataset = 'msd_liver_seg_test_down2'
#     im_channel = 1
#     class_num = 3
#     normalization = None #'softmax'  # To use bce loss.
#
#     # model
#     optimizer = 'adam'
#     lr = 1e-2
#     scheduler = 'StepLR_multi_step'
#     learning_rate_decay = [150, 350] # [150, 300] only for wo skip
#     patience = 20
#     verbose = 1
#     train_batch = 4
#     train_patch_fg_ratio = 0.65
#     val_batch = 1
#     val_freq = 5
#     save_model_freq = 50
#     num_workers = 1
#     max_queue_size = num_workers * 1
#     epochs = 10000
#     dice_weight = [0.5, 1, 10]
#     bce_weight = 1
#     loss = 'nnunet_ce_dice3'
#     save_val_results = True
#     # pretrain
#     resume = None #'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220531-123611/latest_model.pth' # '../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220531-001039/best_model.pth'#'../checkpoints/msd_liver_seg_train_down2/unet_3d_Simple_msd_liver_seg_fs_nnunet_stage0/20220530-155939/best_model.pth'
#     pretrained_model = None# '../checkpoints/luna_rcb_pretask/unet_3d_rcb_Simple_RCB_ct/20220418-083822/RCB_CT_473.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_rcb_om_pretask/unet_3d_rcb_om_Simple_RCB_OM_ct/20220510-203611/RCB_CT_532.pth' #'../checkpoints/luna_rcb_pretask/unet_3d_rcb_Simple_RCB_ct/20220418-083822/RCB_CT_473.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_mg_pretask/unet_3d_wo_skip_Simple_genesis_chest_ct/20220423-145224/Genesis_Chest_CT_66.pth' #'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'#'../checkpoints/luna_ssm_rot_pretask/SSM_Rot_CT_64.pth'#'../checkpoints/luna_pcrl_pretask/mypcrl_100.pth'#'../checkpoints/luna_pcrl_pretask/pcrl_3d_PCRL_Model_mypcrl_luna/20220329-162814/100.pth' #'../checkpoints/luna_pcrl_pretask/pcrl_3d_EMA_Model_pcrl_luna/20220322-190451/65.pth'#'../checkpoints/luna_mg_pretask/unet_3d_Simple_genesis_chest_ct/20220311-161634/Genesis_Chest_CT_250.pth'
#     transferred_part = 'all'
#     ## pretrained model keys: transferred_dismatched_keys[0]; fine-tuned model keys: transferred_dismatched_keys[1]
#     transferred_dismatched_keys = ['module.encoder.', 'module.']
#
#     def display(self, logger):
#         """Display Configuration values."""
#         logger.info("\nConfigurations:")
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
#                 logger.info("{:30} {}".format(a, getattr(self, a)))
#         logger.info("\n")
#

if __name__ == '__main__':
    config = msdliver_config()
    Trainer = Seg3DMCTrainer(config)
    Trainer.train()






