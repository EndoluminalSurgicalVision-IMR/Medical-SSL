# Medical-SSL

### [NEWS-20230417]
We have added the configs of 2D pretraining and fine-tuning with EyePACS and DRIVE dataset. Please refer to "configs_2d"

### The code of our paper: 

Chuyan Zhang, Hao Zheng, Yun Gu, "Dive into Self-Supervised Learning for Medical Image Analysis"

### How to run?
To run the benchmark, please refer to the config files in "configs/"

### Dependencies
*	Python 3.7
*	PyTorch 1.7.1

### How to perform pretraining?

**Step1. Prepare the pretraining dataset** 

Download the LUNA2016 from https://luna16.grand-challenge.org/download/

Store the LUNA2016 dataset in the path "../../Data/LUNA2016"

**Step2. Pre-process the pretraining data for different pretext tasks.** 

Pre-Process the LUNA2016 dataset by the code in the fold  pre_processing:

* Predictive SSL: RPL/ROT/Jigsaw/RKB/ RKB+ pretext tasks
  
  ```python preprocess_luna_ssm.py```

*  Generative SSL: MG/AE pretext tasks
  
    ```python -W ignore infinite_generator_3D.py --fold $subset --scale 32 --data ../../Data/LUNA2016 --save generated_cubes```
   
*   Contrastive SSL: PCRL/SimCLR/BYOL
  
    ```python luna_pcrl_generator.py --input_rows 64 --input_cols 64 --input_deps 32 --data ../../Data/LUNA2016 --save processedLUNA_save_path```

**Step3. List the paths to the pre-processed datasets in datasets_3D/paths.py** 

**Step4. Pretrain the pretxt tasks.** 

Find the corresponding config files to different SSL pretext tasks in "configs/", make sure the configs match your training setting:

 ```python configs/luna_xxx_3d_config.py```


### How to fine-tuning?

**Step1. Prepare the target dataset** 

**Step2. Pre-process the target dataset** 
Example: For data processing in NCC task:
 ```python luna_node_extraction.py```

**Step3. List the paths to the pre-processed datasets in datasets_3D/paths.py** 

**Step4. Fine-tune a pretrained model on the target dataset.** 

Find the corresponding config files to target tasks in "configs/" or write your own , make sure the configs match your training setting:


### On going...
We are still working on more implementations of self-supervised methods for medical image. Feel free to contribute!

### More?
The full paper can be found [here](https://arxiv.org/pdf/2209.12157). More details can be found in the [supplementary material](appendix.pdf).






