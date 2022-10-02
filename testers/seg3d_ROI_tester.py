from testers.base_tester import BaseTester
from utils.tools import *
from tqdm import tqdm
from utils.metrics import SegMetric_Numpy, dice, iou

"""
config
-- gpu_ids
-- test_dataset
-- network
-- model_path
-- save_results_path
"""


class Seg3DROITester(BaseTester):
    def __init__(self, config):
        super(Seg3DROITester, self).__init__(config)

    def test_all_cases(self):
        self.network.eval()
        with torch.no_grad():
            self.logger.info("validating....")
            gts = []
            preds = []
            for itr, (image, gt, index) in tqdm(enumerate(self.test_dataloader)):
                image = image.to(self.device)
                gt = gt.to(self.device)
                pred = self.network(image)
                # to numpy
                gt = gt.cpu().int().numpy()
                pred = pred.cpu().numpy()
                # pred = np.where(pred>0.5, 1, 0)
                gts.extend(gt)
                preds.extend(pred)

            gts = np.array(gts).squeeze()
            preds = np.array(preds).squeeze()

            assert len(preds.shape) == 4 and len(gts.shape) == 4

            # calculate the dice and iou
            eval_metric = SegMetric_Numpy()
            DCS_all = []
            IOU_all = []
            for i in tqdm(range(gts.shape[0])):
                eval_metric.update(preds[i], gts[i])
                acc, SE, SP, PC, F1, js, dc = eval_metric.get_current
                DCS_all.append(dc)
                IOU_all.append(js)

            dice_avg, iou_avg = np.average(DCS_all), np.average(IOU_all)
            acc, SE, SP, PC, F1, js, dc = eval_metric.get_avg

            print("y:  {} | {:.1f} ~ {:.1f}".format(gts.shape, np.min(gts), np.max(gts)))
            print("p:  {} | {:.1f} ~ {:.1f}".format(preds.shape, np.min(preds), np.max(preds)))
            print("[ALL]  Dice = {:.2f}%".format(100.0 * dice(preds, gts)))
            print("[ALL]  IoU  = {:.2f}%".format(100.0 * iou(preds, gts)))
            print("[AVG] Dice = {:.2f}%".format(100.0 * dice_avg))
            print("[AVG] IoU  = {:.2f}%".format(100.0 * iou_avg))
            print("[AVG_avg] Dice = {:.2f}%".format(100.0 * dc))
            print("[AVG_avg] IoU  = {:.2f}%".format(100.0 * js))




