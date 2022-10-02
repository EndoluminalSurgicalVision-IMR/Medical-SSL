from testers.base_tester import BaseTester
from utils.tools import *
from tqdm import tqdm
from sklearn import metrics
import pandas as pd


def get_bn_statis(model):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'running_mean' in name:
            means.append(param.clone())
        elif 'running_var' in name:
            vars.append(param.clone())
    return means, vars


def cal_distance(means_1, means_2, vars_1, vars_2):
    pdist = torch.nn.PairwiseDistance(p=2)
    dis = 0
    for (mean_1, mean_2, var_1, var_2) in zip(means_1, means_2, vars_1, vars_2):
        dis += (pdist(mean_1.reshape(1, mean_1.shape[0]), mean_2.reshape(1, mean_2.shape[0])) + pdist(var_1.reshape(1, var_1.shape[0]), var_2.reshape(1, var_2.shape[0])))
    return dis.item()

class Classification2CTester(BaseTester):
    def __init__(self, config):
        super(Classification2CTester, self).__init__(config)

    def test_all_cases(self):
        gts = []
        preds = []
        indexs = []
        means_train, vars_train = get_bn_statis(self.network)
        print('mean vars')
        bn_dis_avg = []
        self.network.eval()
        with torch.no_grad():
            for step, (input, target, image_name) in tqdm(enumerate(self.test_dataloader)):
                # self.network = self.network.eval()
                # print('training', self.network.training)
                input = input.to(self.device)
                target = target.to(self.device)
                pred = self.network(input)

                means_test, vars_test = get_bn_statis(self.network)
                new_dis = cal_distance(means_test, means_train, vars_test, vars_train)
                self.logger.info('****************bn_dis {} ***********'.format(new_dis))
                bn_dis_avg.append(new_dis)

                image_array = input.cpu().numpy()
                target_array = target.cpu().numpy()
                pred_array = pred.cpu().numpy()
                gts.extend(target_array)
                preds.extend(pred_array)
                indexs.extend(image_name)
                print(target_array)
                print(np.where(pred_array>=0.5, 1, 0))
                print(
                    "input:  {} | {:.1f} ~ {:.1f}".format(image_array.shape, np.min(image_array), np.max(image_array)))
                print("gt:  {} | {:.1f} ~ {:.1f}".format(target_array.shape, np.min(target_array), np.max(target_array)))
                print("pred:  {} | {:.1f} ~ {:.1f}".format(pred_array.shape, np.min(pred_array), np.max(pred_array)))
                # if step == 2:
                #     break
        # ROC curve
        print('index', len(indexs))
        gts = np.array(gts, dtype=np.int32).squeeze()
        preds = np.array(preds, dtype=np.float32).squeeze()

        data_frame = pd.DataFrame(
            data={'name': indexs, 'gts': gts, 'preds': preds},
            index=range(gts.shape[0]))
        data_frame.to_csv(self.config.save_results_path + '/results.csv', index_label='Index')

        self.logger.info("gts:  {} | {:.1f} ~ {:.1f}".format(gts.shape, np.min(gts), np.max(gts)))
        self.logger.info("preds:  {} | {:.1f} ~ {:.1f}".format(preds.shape, np.min(preds), np.max(preds)))
        fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
        self.logger.info("[EVAL] AUC = {:.2f}%".format(100.0 * metrics.auc(fpr, tpr)))
        self.logger.info(" BN dis: {:.6f}".format(np.average(bn_dis_avg)))

        # Confusion matrix
        preds_argmax = np.array(preds > 0.5).astype(np.int16)
        gts_argmax = np.array(gts > 0.5).astype(np.int16)
        confusion_matrix = metrics.confusion_matrix(gts_argmax, preds_argmax, sample_weight=None)
        self.logger.info("[EVAL] Confusion_Matrix = {}%".format(confusion_matrix))
