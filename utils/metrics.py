import threading
import torch
import numpy as np
import torch.nn.functional as F

class SegMetric_Numpy(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.SE = 0.0
        self.SP = 0.0
        self.PC = 0.0
        self.F1 = 0.0
        self.JS = 0.0
        self.DC = 0.0
        self.length = 0.0

        self.acc_i = 0.0
        self.SE_i = 0.0
        self.SP_i = 0.0
        self.PC_i = 0.0
        self.F1_i = 0.0
        self.JS_i = 0.0
        self.DC_i = 0.0

    def update(self, SR, GT, threshold=0.5):
        """
            SR:  Segmentation Result, a ndarray with the shape of [D, H, W]
            GT:  Ground Truth , a ndarray with the shape of [D, H, W]

        return:
            acc, SE, SP, PC, F1, js, dc
        """
        assert SR.shape == GT.shape

        SR = (SR > threshold).astype(int)
        GT = (GT == np.max(GT)).astype(int)
        corr = np.sum(SR == GT)
        tensor_size = SR.shape[0] * SR.shape[1] * SR.shape[2]
        self.acc_i = float(corr) / float(tensor_size)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR == 1) & (GT == 1))
        FN = ((SR == 0) & (GT == 1))
        # TN : True negative
        # FP : False Positive
        TN = ((SR == 0) & (GT == 0))
        FP = ((SR == 1) & (GT == 0))

        self.SE_i = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)
        self.SP_i = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)
        self.PC_i = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)
        self.F1_i = 2 * self.SE_i * self.PC_i / (self.SE_i + self.PC_i + 1e-6)

        Inter = np.sum((SR + GT) == 2)
        Union = np.sum((SR + GT) >= 1)

        print('inter union', Inter, Union)
        self.JS_i = float(Inter) / (float(Union) + 1e-6)
        self.DC_i = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

        self.acc += self.acc_i
        self.SE += self.SE_i
        self.SP += self.SP_i
        self.PC += self.PC_i
        self.F1 += self.F1_i
        self.JS += self.JS_i
        self.DC += self.DC_i
        self.length += 1

    @property
    def get_current(self):
        return self.acc_i, self.SE_i, self.SP_i, self.PC_i, self.F1_i, self.JS_i, self.DC_i

    @property
    def get_avg(self):
        return self.acc / self.length, self.SE / self.length, \
               self.SP / self.length, self.PC / self.length, self.F1 / self.length, \
               self.JS / self.length, self.DC / self.length


class SegMetric_Tensor(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.SE = 0.0
        self.SP = 0.0
        self.PC = 0.0
        self.F1 = 0.0
        self.JS = 0.0
        self.DC = 0.0
        self.length = 0.0

        self.acc_i = 0.0
        self.SE_i = 0.0
        self.SP_i = 0.0
        self.PC_i = 0.0
        self.F1_i = 0.0
        self.JS_i = 0.0
        self.DC_i = 0.0

    def update(self, SR, GT, threshold=0.5):
        """
        SR:  Segmentation Result, a tensor with the size of [D, H, W]
        GT:  Ground Truth , a tensor with the size of [D, H, W]

        return:
            acc, SE, SP, PC, F1, js, dc
        """
        assert SR.size() == GT.size()
        SR = (SR > threshold).int()
        GT = (GT == torch.max(GT)).int()
        corr = torch.sum(SR == GT)
        tensor_size = SR.numel()

        # Acc: accuracy
        self.acc_i = float(corr) / float(tensor_size)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR == 1) & (GT == 1))
        FN = ((SR == 0) & (GT == 1))
        # TN : True negative
        # FP : False Positive
        TN = ((SR == 0) & (GT == 0))
        FP = ((SR == 1) & (GT == 0))

        self.SE_i = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
        self.SP_i = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
        self.PC_i = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
        self.F1_i = 2 * self.SE_i * self.PC_i / (self.SE_i + self.PC_i + 1e-6)

        Inter = torch.sum((SR + GT) == 2)
        ## torch.sum((SR + GT) == 2) == torch.sum(((SR + GT) == 2).int()) is True

        Union = torch.sum((SR + GT) >= 1)

        self.JS_i = float(Inter) / (float(Union) + 1e-6)
        self.DC_i = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

        self.acc += self.acc_i
        self.SE += self.SE_i
        self.SP += self.SP_i
        self.PC += self.PC_i
        self.F1 += self.F1_i
        self.JS += self.JS_i
        self.DC += self.DC_i
        self.length += 1

    @property
    def get_current(self):
        return self.acc_i, self.SE_i, self.SP_i, self.PC_i, self.F1_i, self.JS_i, self.DC_i

    @property
    def get_avg(self):
        return self.acc / self.length, self.SE / self.length, \
               self.SP / self.length, self.PC / self.length, self.F1 / self.length, \
               self.JS / self.length, self.DC / self.length




def iou(im1, im2):
    overlap = (im1 > 0.5) * (im2 > 0.5)
    union = (im1 > 0.5) + (im2 > 0.5)
    print('overlap', overlap.sum())
    print('union', float(union.sum()))
    return overlap.sum() / float(union.sum())


def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    print('ier sum', intersection.sum(), im_sum)

    return 2. * intersection.sum() / im_sum


