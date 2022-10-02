from testers.base_tester import BaseTester
from utils.tools import *
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import SegMetric_Numpy
import skimage.measure as measure
import skimage.morphology as morphology
import copy
import cv2
from utils.metrics import iou, dice
import pandas as pd

"""
config
-- gpu_ids
-- test_dataset
-- network
-- model_path
-- save_results_path
-- cut_params
"""


class Seg3DTester(BaseTester):
    def __init__(self, config):
        super(Seg3DTester, self).__init__(config)

        if self.config.normalization is None:
            if self.config.class_num == 1:
                self.pred_norm = lambda x: F.sigmoid(x)
            else:
                self.pred_norm = lambda x: F.softmax(x, 1)
        else:
            self.pred_norm = lambda x: x

    def test_all_cases(self):

        if self.config.class_num == 1:
            # Only calculate the dice of one class
            eval_metric = SegMetric_Numpy()
            Index_all = []
            DCS_all = []
            IoU_all = []
            self.network.eval()
            for iter, sample in tqdm(enumerate(self.test_dataloader)):

                # sample: patches [B=1, N, C, D, H, W], label [B=1, K, D, H, W], image_info: org_shape, new_shape, image_index
                pred_aggregated, label, image_index = self.test_one_case(sample)
                # pred_aggregated : nd.array [K, D, H, W] label: nd.array [K, D, H, W]
                print('************ pred agg',  pred_aggregated.shape, label.shape)
                eval_metric.update(pred_aggregated[0], label[0])
                Index_all.append(image_index)
                # gts.extend(list(label[0].flatten()))
                # preds.extend(list((pred_aggregated[0].flatten())))
                acc, SE, SP, PC, F1, js, dc = eval_metric.get_current
                self.logger.info("Test cur-patient {} dice:{:.4f} iou:{:.4f} acc:{:.4f} SE:{:.4f} SP:{:.4f} F1:{:.4f}"
                                 " PC:{:.4f}".format(image_index, dc, js, acc, SE, SP, F1, PC))
                DCS_all.append(dice(pred_aggregated[0], label[0]))
                IoU_all.append(iou(pred_aggregated[0], label[0]))

            acc, SE, SP, PC, F1, js, dc = eval_metric.get_avg
            self.logger.info("Test avg dice:{:.4f} iou:{:.4f} acc:{:.4f} SE:{:.4f} SP:{:.4f} F1:{:.4f} PC:{:.4f}"
                             "".format(dc, js, acc, SE, SP, F1, PC))
            self.logger.info(" max dice:{:.4f} min dice:{:.4f} ".format(max(DCS_all), min(DCS_all)))

            # gts = np.array(gts)
            # preds = np.array(preds)
            # print(gts.shape, preds.shape)
            # assert len(preds.shape) == 1 and len(gts.shape) == 1


            # print("y:  {} | {:.1f} ~ {:.1f}".format(gts.shape, np.min(gts), np.max(gts)))
            # print("p:  {} | {:.1f} ~ {:.1f}".format(preds.shape, np.min(preds), np.max(preds)))
            # print("[ALL]  Dice = {:.2f}%".format(100.0 * dice(preds, gts)))
            # print("[ALL]  IoU  = {:.2f}%".format(100.0 * iou(preds, gts)))
            print("[AVG] Dice = {:.2f}%".format(100.0 * np.average(DCS_all)))
            print("[AVG] IoU  = {:.2f}%".format(100.0 * np.average(IoU_all)))

            # save results
            Index_org = copy.deepcopy(Index_all)
            Index_all.sort()
            order_index = [j for i in range(len(Index_all)) for j in range(len(Index_org)) if Index_all[i] == Index_org[j]]
            DCS_all = [ DCS_all[i] for i in order_index]
            data_frame = pd.DataFrame(
                data={'Case': Index_all, 'Dice': DCS_all}, index=range(len(Index_all)))
            data_frame.to_csv(self.config.save_results_path + '/results.csv',
                              index_label='Index')


        else:
            # Calculate the dice of different class, respectively.
            eval_metrics = []
            Save_all = {}
            for class_i in range(self.config.class_num - 1):
                eval_metric_i = SegMetric_Numpy()
                eval_metrics.append(eval_metric_i)
                Save_all['index_class_' + str(class_i)] = []
                Save_all['dice_class_' + str(class_i)] = []
            self.network.eval()
            for step, sample in enumerate(self.test_dataloader):
                pred_aggregated, label, image_index = self.test_one_case(sample)
                for class_i in range(self.config.class_num - 1):
                    # pred_aggregated : nd.array [K, D, H, W] label: nd.array [K, D, H, W]
                    pred_array = pred_aggregated[class_i + 1]
                    target_array = label[class_i + 1]
                    if np.sum(target_array > 0.5) > 30:
                        print('target_array', np.min(target_array), np.max(target_array), np.sum(target_array))
                        eval_metrics[class_i].update(pred_array, target_array)
                        acc, SE, SP, PC, F1, js, dc = eval_metrics[class_i].get_current
                        self.logger.info(
                            "Test cur-patient {}  class {} dice:{:.4f} iou:{:.4f} acc:{:.4f} SE:{:.4f} SP:{:.4f} F1:{:.4f} PC:{:.4}"
                            .format(image_index, class_i+1, dc, js, acc, SE, SP, F1, PC))
                        Save_all['index_class_' + str(class_i)].append(image_index)
                        Save_all['dice_class_' + str(class_i)].append(dc)
                    else:
                        self.logger.info(
                            "Test cur-patient {}  class {}***No target region***".format(image_index, class_i + 1))

            dice_avg_all_classes = 0.0
            iou_avg_all_classes = 0.0
            for class_i in range(self.config.class_num - 1):
                acc, SE, SP, PC, F1, js, dc = eval_metrics[class_i].get_avg
                self.logger.info(" Test Case_Avg Class {} dice:{:.4f} iou:{:.4f} acc:{:.4f} SE:{:.4f} SP:{:.4f}"
                                          " F1:{:.4f} PC:{:.4f} ".format(class_i, dc, js, acc, SE, SP, F1, PC))

                dice_avg_all_classes += dc
                iou_avg_all_classes += js

                # save results of each class
                data_frame = pd.DataFrame(
                    data={'Case': Save_all['index_class_' + str(class_i)], 'Dice': Save_all['dice_class_' + str(class_i)]},
                    index=range(len(Save_all['index_class_' + str(class_i)])))
                data_frame.to_csv(self.config.save_results_path + '/class_' + str(class_i) + '_results.csv', index_label='Index')

            dice_avg_all_classes = dice_avg_all_classes / (self.config.class_num - 1)
            iou_avg_all_classes = iou_avg_all_classes / (self.config.class_num - 1)

            self.logger.info(" Test Case_Avg Class_Avg dice:{:.4f} iou:{:.4f}".format(dice_avg_all_classes, iou_avg_all_classes))

    def test_one_case(self, sample):
        # sample: patches [B=1, N, C, D, H, W], label [B=1, K, D, H, W], image_info: org_shape, new_shape, image_index
        image, full_image, patches, label, image_info = sample

        patches = patches[0].to(self.device)
        org_shape = image_info['org_shape'][0].numpy()
        new_shape = image_info['new_shape'][0].numpy()
        image_index = image_info['image_index'][0]

        self.aggregater = Aggregate_3DSeg_tool(num_classes=self.config.class_num,
                                               img_org_shape=org_shape,
                                               img_new_shape=new_shape,
                                               C=self.config.test_cut_params)
        for i in tqdm(range(patches.size()[0])):
            with torch.no_grad():
                patch = patches[i].unsqueeze(0)
                # patch [B=1, C, pd, ph, pw]
                pred = self.network(patch).cpu()
                pred = self.pred_norm(pred)
                # pred [B=1, K, pd, ph, pw]
                print('pred-----', pred.size())
                # pred = np.where(pred > 0.5, 1, 0)
                self.aggregater.add_patch_result(pred)

                # save_patch_predictions_path = os.path.join(self.save_results_path, 'test_patch_results/'+image_index)
                #
                # # save patch predictions
                # patch_cur = patch[0][0].cpu().numpy()
                # pred_cur = pred[0][0].cpu().int().numpy()
                # pred_cur = np.where(pred_cur > 0.5, np.ones_like(pred_cur), np.zeros_like(pred_cur))
                # #pred_cur = one_hot_reverse(pred[0][0].cpu().numpy())
                # save_np2nii(patch_cur, save_patch_predictions_path, 'patch' + str(i))
                # print('saved patch image shape', patch_cur.shape)
                # save_np2nii(pred_cur, save_patch_predictions_path, 'patch_pre' + str(i))
                # print('saved patch pred shape', pred_cur.shape)

        pred_aggregated = self.aggregater.recompone_overlap()

        if self.config.class_num == 1:
            pred_aggregated = torch.where(pred_aggregated > 0.5, 1, 0).int()
        else:
            pred_aggregated = torch.argmax(pred_aggregated, 0).int()
            pred_aggregated = self.create_one_hot_label(pred_aggregated)

        # pred: [K, D, H, W]

        pred_aggregated = pred_aggregated.cpu().numpy()
        if self.config.object == 'liver' and self.config.post_processing:
            for i in range(pred_aggregated.shape[0]):
                pred_aggregated[i] = self.post_processing(pred_aggregated[i])
        label = label[0].int().cpu().numpy()

        # save images
        save_predictions_path = os.path.join(self.save_results_path, 'test_results')

        image_cur = image[0][0].cpu().numpy()
        # full_image_cur = full_image[0][0].cpu().numpy()
        label_all_classes = one_hot_reverse(label)
        pred_aggregated_all_classes = one_hot_reverse(pred_aggregated)

        save_np2nii(pred_aggregated_all_classes, save_predictions_path, 'pre' + image_index)
        print('saved aggregate pred shape', pred_aggregated_all_classes.shape)
        save_np2nii(image_cur, save_predictions_path, 'img' + image_index)
        print('saved image shape', image_cur.shape)
        # save_np2nii(full_image_cur, save_predictions_path, 'padding_img' + image_index)
        # print('saved full image shape', full_image_cur.shape)
        save_np2nii(label_all_classes, save_predictions_path, 'label' + image_index)
        print('saved label shape', label_all_classes.shape)

        return pred_aggregated, label, image_index

    def post_processing(self, pred_seg):
        pred_seg = pred_seg.astype(np.uint8)
        liver_seg = copy.deepcopy(pred_seg)
        liver_seg = measure.label(liver_seg, background=0, connectivity=3)
        props = measure.regionprops(liver_seg)

        max_area = 0
        max_index = 0
        for index, prop in tqdm(enumerate(props, start=1)):
            if prop.area > max_area:
                max_area = prop.area
                max_index = index

        liver_seg[liver_seg != max_index] = 0
        liver_seg[liver_seg == max_index] = 1

        liver_seg = liver_seg.astype(np.bool)
        morphology.remove_small_holes(liver_seg, 5e4, connectivity=2, in_place=True)
        liver_seg = liver_seg.astype(np.uint8)
        # print('remove holes', np.min(liver_seg), np.max(liver_seg))
        return liver_seg

    def close_operator(self, pred_seg):
        pred_seg = pred_seg.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        pred_seg = cv2.morphologyEx(pred_seg, cv2.MORPH_CLOSE, kernel)
        pred_seg = np.array(pred_seg).astype(np.uint8)
        return pred_seg

    def create_one_hot_label(self, label):
        """
        Inputlabel:[D,H,W].
        Outputlabel:[K,D,H,W].The output label contains the background class in the 0th channel.
        """
        onehot_label = torch.zeros(
            (self.config.class_num, label.shape[0], label.shape[1], label.shape[2]))
        for i in range(self.config.class_num):
            onehot_label[i, :, :, :] = (label == i).int()

        return onehot_label









