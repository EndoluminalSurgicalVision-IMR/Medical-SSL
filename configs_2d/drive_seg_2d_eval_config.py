import os
from testers import *
from models import models_dict
from networks import networks_dict
from datasets_2D import datasets_dict_2D
import argparse
models_name = models_dict.keys()
datasets_name = datasets_dict_2D.keys()
networks_name = networks_dict.keys()

parser = argparse.ArgumentParser(description='2D_Unet Test')
# Add default argument
parser.add_argument('--gpu_ids', type=int,default=[0, 1], help="For example 0,1 to run on two GPUs")
parser.add_argument('--network',  type=str, default='unet_2d',choices=networks_name,
                    help='Networks')
parser.add_argument('--init_weight_type', type=str, choices=["kaiming",'normal','xavier','orthogonal'],
                    default="kaiming", help=" model init mode")
parser.add_argument('--note', type=str, default='baseline_from_scratch',
                    help='model note ')
parser.add_argument('--eval_dataset', type=str, default='drive_seg',choices=datasets_name,
                    help='val data')
parser.add_argument('--im_channel', type=int, default=3, help="input image channel ")
parser.add_argument('--out_ch', type=int, default=1, help="input image channel ")
parser.add_argument('--class_num', type=int, default=1, help="output feature channel")
parser.add_argument('--normalization', type=str, choices=['sigmoid', 'softmax', 'none'],
                    default=None, help="the normalization of output")
parser.add_argument('--val_batch', type=int, default=2, help="val_batch")
parser.add_argument('--num_workers', type=int, default=0, help="dataloader numworkers")
parser.add_argument('--model_path', type=str, default=None) 
parser.add_argument('--save_results_path', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    Tester =  Seg2DROITester(args)
    Tester.test_all_cases()


