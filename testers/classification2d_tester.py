from testers.base_tester import BaseTester
from utils.tools import *
from tqdm import tqdm
from sklearn import metrics
import pandas as pd
from utils.metrics import ClsEstimator


class Classification2DTester(BaseTester):
    def __init__(self, config):
        super(Classification2DTester, self).__init__(config)
        self.estimator = ClsEstimator(criterion=self.config.loss, num_classes=5, labels=['0', '1', '2', '3', '4'])

    def test_all_cases(self):
        gts = []
        preds = []
        indexs = []
        self.network.eval()
        with torch.no_grad():
            for step, (input, target, image_name) in tqdm(enumerate(self.test_dataloader)):
                input = input.to(self.device)
                target = target.to(self.device)
                pred = self.network(input)
                if self.config.loss == 'ce':
                    pred = torch.softmax(pred, dim=1)
                self.estimator.update(pred, target)
                # self.confusion.update(pred, target)
                image_array = input.cpu().numpy()
                target_array = target.cpu().numpy()
                pred_array = pred.cpu().numpy()
                gts.extend(target_array)
                preds.extend(pred_array)
                indexs.extend(image_name)
                print("input:  {} | {:.1f} ~ {:.1f}".format(image_array.shape, np.min(image_array), np.max(image_array)))
                print("gt:  {} | {:.1f} ~ {:.1f}".format(target_array.shape, np.min(target_array), np.max(target_array)))
                print("pred:  {} | {:.1f} ~ {:.1f}".format(pred_array.shape, np.min(pred_array), np.max(pred_array)))

        # ROC curve
        acc = self.estimator.get_accuracy(-1)
        kappa = self.estimator.get_kappa(-1)
        self.estimator.summary()
        self.logger.info("[EVAL] ACC = {:.2f}%".format(acc*100))
        self.logger.info("[EVAL] Weighted-KAPPA = {:.2f}%".format(kappa*100))
        self.estimator.plot(self.save_results_path)

        gts = np.array(gts, dtype=np.int32).squeeze()
        preds = np.array(preds, dtype=np.float32).squeeze()

        data_frame = pd.DataFrame(
            data={'name': indexs, 'gts': gts, 'preds': preds},
            index=range(gts.shape[0]))
        data_frame.to_csv(self.config.save_results_path + '/results.csv', index_label='Index')
