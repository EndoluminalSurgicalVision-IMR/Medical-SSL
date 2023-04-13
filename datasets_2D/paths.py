# Save the paths of dataset dirs.
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'eyepacs_cls_per10':
            return '../data/Kaggle/per10_split_gradable'

        elif dataset == 'eyepacs_mg_pretask' or dataset=='eyepacs_cl_pretask'  \
                 or dataset == 'eyepacs_pcrl_pretask':
            return '../data/Kaggle/MG_gradable'

        elif dataset == 'eyepacs_rot_pretask' or dataset == 'eyepacs_rpl_pretask' or \
                dataset == 'eyepacs_jigsaw_pretask' or dataset=='eyepacs_ae_pretask':
            return '../data/Kaggle/MG_gradable_right'
        
        elif dataset == 'drive_seg':
            return '../data/DRIVE'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError



