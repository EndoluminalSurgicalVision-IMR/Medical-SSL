from testers.base_tester import BaseTester
from utils.tools import *
from tqdm import tqdm
from utils.metrics import SegMetric_Numpy, dice, iou, mean_iou, cal_mean_iou, mean_iou_org
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


class Seg2DROITester(BaseTester):
    def __init__(self, config):
        super(Seg2DROITester, self).__init__(config)

    def test_all_cases(self):
        self.network.eval()
        with torch.no_grad():
            self.logger.info("validating....")
            gts = []
            preds = []
            masks = []
            for itr, (image, gt, mask, index) in tqdm(enumerate(self.test_dataloader)):
                image = image.to(self.device)
                gt = gt.to(self.device)
                mask = mask.to(self.device)
                pred = self.network(image)
                # to numpy
                gt = gt.cpu().int().numpy()
                mask = mask.cpu().int().numpy()
                pred = pred.cpu().numpy()

                for j in range(image.shape[0]):
                    save_tensor2image(image[j],
                                      path=self.save_results_path+ '/test/' + index[j],
                                      name='input')
                    save_np2image(np.squeeze(gt[j]),
                                  path=self.save_results_path+ '/test/' + index[j],
                                  name='label')
                    save_np2image(np.squeeze(np.where(pred[j] > 0.5, 1, 0)),
                                  path=self.save_results_path+ '/test/' + index[j],
                                  name='pred')

                gts.extend(gt)
                preds.extend(pred)
                masks.extend(mask)

            # get all the data
            gts = np.array(gts).squeeze()
            preds = np.array(preds).squeeze()
            masks = np.array(masks).squeeze()

            # calculate the dice and iou
            eval_metric = SegMetric_Numpy()
            eval_metric_in_fov = SegMetric_Numpy()
            DCS_all = []
            DSC_FOV_all = []
            ACC_all = []
            ACC_FOV_all = []
            SEN_all = []
            SEN_FOV_all = []
            SPE_all = []
            SPE_FOV_all = []
            for i in tqdm(range(gts.shape[0])):
                eval_metric.update(preds[i], gts[i])
                eval_metric_in_fov.update_in_FOV(preds[i], gts[i], masks[i])
                acc, SE, SP, PC, F1, js, dc = eval_metric.get_current
                acc2, SE2, SP2, PC2, F12, js2, dc2 = eval_metric_in_fov.get_current
                DCS_all.append(dc)
                DSC_FOV_all.append(dc2)
                ACC_all.append(acc)
                ACC_FOV_all.append(acc2)
                SEN_all.append(SE)
                SEN_FOV_all.append(SE2)
                SPE_all.append(SP)
                SPE_FOV_all.append(SP2)

            acc, SE, SP, PC, F1, js, dc = eval_metric.get_avg
            acc2, SE2, SP2, PC2, F12, js2, dc2 = eval_metric_in_fov.get_avg

            print("[AVG] Dice = {:.2f}%".format(100.0 * dc))
            print("[AVG] Dice-In-FOV  = {:.2f}%".format(100.0 * dc2))
            print("[AVG] ACC  = {:.2f}%".format(100.0 * acc))
            print("[AVG] ACC-In-FOV  = {:.2f}%".format(100.0 * acc2))

            data_frame = pd.DataFrame(
                data={'Dice': DCS_all, 'Dice-FOV': DSC_FOV_all,
                      'ACC': ACC_all, 'ACC-FOV':ACC_FOV_all,
                      'SEN': SEN_all, 'SEN-FOV': SEN_FOV_all,
                      'SPE': SPE_all, 'SPE-FOV': SPE_FOV_all,
                      },
                index=range(gts.shape[0]))
            data_frame.to_csv(self.config.save_results_path + '/results.csv', index_label='Index')

            self.logger.info("[AVG] Dice = {:.2f}%".format(100.0 * np.average(DCS_all)))
            self.logger.info("[AVG] ACC  = {:.2f}%".format(100.0 * np.average(ACC_all)))
            self.logger.info("[STD] Dice = {:.2f}%".format(100.0 * np.std(DCS_all)))
            self.logger.info("[STD] ACC  = {:.2f}%".format(100.0 * np.std(ACC_all)))

            self.logger.info("[AVG] Dice-In-FOV = {:.2f}%".format(100.0 * np.average(DSC_FOV_all)))
            self.logger.info("[AVG] ACC-In-FOV  = {:.2f}%".format(100.0 * np.average(ACC_FOV_all)))
            self.logger.info("[STD] Dice-In-FOV = {:.2f}%".format(100.0 * np.std(DSC_FOV_all)))
            self.logger.info("[STD] ACC-In-FOV  = {:.2f}%".format(100.0 * np.std(ACC_FOV_all)))
            logging.shutdown()






