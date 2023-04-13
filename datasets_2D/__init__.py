from datasets_2D.Classification.eyepacs_classification import ClassificationEyePACSSet
from datasets_2D.MG.eyepacs_mg_pretask import MGEyepacsPretaskSet
from datasets_2D.MG.eyepacs_ae_pretask import AEEyepacsPretaskSet
from datasets_2D.MG.eyepacs_pcrl_pretask import PCRLEyepacsPretaskSet
from datasets_2D.CL.eyepacs_CL_pretask import CLEyepacsPretaskSet
from datasets_2D.PTP.eyepacs_rot_pretask import RotEyepacsPretaskSet
from datasets_2D.PTP.eyepacs_rpl_pretask import RPLEyepacsPretaskSet
from datasets_2D.Jigsaw.eyepacs_jigsaw_pretask import JigSawEyepacsPretaskSet
from datasets_2D.Seg.drive_segmentation import SegDRIVESet
from datasets_2D.paths import Path
from torch.utils.data import DataLoader, WeightedRandomSampler


datasets_dict_2D = {
    'eyepacs_cls_per10': ClassificationEyePACSSet
    'eyepacs_mg_pretask': MGEyepacsPretaskSet,
    'eyepacs_ae_pretask':AEEyepacsPretaskSet,
    'eyepacs_cl_pretask': CLEyepacsPretaskSet,
    'eyepacs_pcrl_pretask':PCRLEyepacsPretaskSet,
    'eyepacs_rot_pretask': RotEyepacsPretaskSet,
    'eyepacs_rpl_pretask': RPLEyepacsPretaskSet,
    'eyepacs_jigsaw_pretask':JigSawEyepacsPretaskSet,
    'drive_seg': SegDRIVESet}



def get_dataloder_2D(args, flag="train", drop_last=True):
    '''
    :return: the dataloader of special datasets
    '''

    if flag == "train":
        print('---------------Building training dataloder-------------------')
        datasets_name = args.train_dataset
        assert datasets_name in datasets_dict_2D.keys(), "The dataset use {} is not exist ".format(datasets_name)
        root = Path.db_root_dir(datasets_name)
        dataset = datasets_dict_2D[datasets_name](config=args, base_dir=root, flag=flag)
        batch_size = args.train_batch
        shuffle = True
        num_workers = args.num_workers

    else:
        print('---------------Building test dataloder-------------------')
        datasets_name = args.eval_dataset
        assert datasets_name in datasets_dict_2D.keys(), "The dataset use {} is not exist ".format(datasets_name)
        root = Path.db_root_dir(datasets_name)
        dataset = datasets_dict_2D[datasets_name](config=args, base_dir=root, flag=flag)
        batch_size = args.val_batch
        shuffle = False
        # num_workers = args.num_workers
        num_workers = 0

    ### pytorch dataloader
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=drop_last)

    return dataset, data_loader



