from testers.base_tester import BaseTester
from utils.tools import *
from tqdm import tqdm
from testers.base_tester import *
from sklearn import metrics
import pandas as pd


class MGTester(BaseTester):
    def __init__(self, config):
        super(MGTester, self).__init__(config)
        self.val_criterion = torch.nn.MSELoss(reduction='mean')


    def load_compared_model(self, network_name, checkpoint):
        self.config.network = network_name
        self.network2 = get_networks(self.config).cuda()
        checkpoint = torch.load(checkpoint)
        self.network2 = torch.nn.DataParallel(self.network2, device_ids=self.config.gpu_ids).to(self.device)
        self.network2.load_state_dict(checkpoint['state_dict'], strict=True)
        self.logger.info("Load weight from {}".format(checkpoint))

    def test_all_cases(self):
        gts = []
        preds = []
        val_losses = []
        self.network.eval()
        with torch.no_grad():
            for step, (input, target) in tqdm(enumerate(self.test_dataloader)):
                input = input.to(self.device)
                target = target.to(self.device)
                pred = self.network(input)
                v_loss = self.val_criterion(pred, target)
                val_losses.append(v_loss.item())
                print(v_loss.mean())
                # if step > 2:
                #     break

        self.logger.info("[EVAL] MSE = {:.4f}".format(np.average(val_losses)))

    def compare_val_imgs(self, network2, checkpoint2):
        self.load_compared_model(network2, checkpoint2)
        val_losses1 = []
        val_losses2 = []
        self.network.eval()
        self.network2.eval()
        with torch.no_grad():
            for itr, (input, target) in tqdm(enumerate(self.test_dataloader)):
                input = input.to(self.device)
                target = target.to(self.device)
                pred1 = self.network(input)
                pred2 = self.network2(input)
                v_loss1 = self.val_criterion(pred1, target)
                v_loss2 = self.val_criterion(pred2, target)
                val_losses1.append(v_loss1.item())
                val_losses2.append(v_loss2.item())

                if itr % 10 == 0:
                    input_np = input[0][0].cpu().numpy()
                    pred_np1 = pred1[0][0].cpu().numpy()
                    pred_np2 = pred2[0][0].cpu().numpy()
                    gt_np = target[0][0].cpu().numpy()
                    image_index = str(itr)
                    assert len(input_np.shape) == 3
                    save_path = os.path.join(self.save_results_path, 'test_patch_results')
                    save_np2nii(input_np, save_path, 'image' + image_index)
                    save_np2nii(pred_np1, save_path, 'pre' + image_index)
                    save_np2nii(pred_np2, save_path, 'pre_wo_skip' + image_index)
                    save_np2nii(gt_np, save_path, 'label' + image_index)

        self.logger.info("[EVAL] MSE1 = {:.4f}".format(np.average(val_losses1)))
        self.logger.info("[EVAL] MSE2 = {:.4f}".format(np.average(val_losses2)))

