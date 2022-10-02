from torch.utils.data import Dataset


class CLBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.crop_size = config.input_size
        self._base_dir = base_dir
        self.all_images = []
        self.flag = flag

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

