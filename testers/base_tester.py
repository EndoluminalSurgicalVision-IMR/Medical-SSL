import torch
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from datasets_3D import get_dataloder_3D, datasets_dict_3D
from networks import get_networks, networks_dict
from models import get_models, models_dict
from utils.tools import get_logger

"""BaseTester is only for downstream tasks such as segmentation or classification.
config
-- gpu_ids
-- test_dataset
-- network
-- model_path
-- save_results_path

"""


class BaseTester(object):
    def __init__(
            self,
            config):
        """
          Steps:
              1、Init logger.
              2、Init device.
              3、Init data_loader.
              4、Init model.
              5、Load model weights from model_path.
              6、Mount the model onto the device (gpus) and set nn.parallel if necessary.

          After this call,
              All will be prepared for testing.
          """
        self.config = config
        self.save_results_path = self.config.save_results_path
        self.logger = get_logger(config.save_results_path)
        self.init()

    def init(self):
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.network = get_networks(self.config).cuda()
        self.device = torch.device('cuda:%d' % self.config.gpu_ids[0]
                                   if torch.cuda.is_available() and len(self.config.gpu_ids) > 0 else 'cpu')

        checkpoint = torch.load(self.config.model_path)
        self.network = torch.nn.DataParallel(self.network, device_ids=self.config.gpu_ids).to(self.device)
        self.network.load_state_dict(checkpoint['state_dict'], strict=True)

        self.logger.info("Load weight from {}".format(self.config.model_path))
        self.test_dataset, self.test_dataloader = get_dataloder_3D(self.config, flag="test", drop_last=False)

    def test_all_cases(self):
        self.network.eval()
        pass

    def test_one_case(self, sample):
        pass
