import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F
import numpy as np


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, normalization='sigmoid'):
        super(U_Net, self).__init__()
        nb_filter = [64, 128, 256, 512,1024]
        #nb_filter = [32,64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Up5 = up_conv(ch_in=nb_filter[4], ch_out=nb_filter[3])
        self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1, stride=1, padding=0)

        if normalization == 'sigmoid':
            assert output_ch == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert output_ch > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.normalization(d1)

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']
        decoder_layers = ['Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Up2', 'Up_conv2']
        out_layers = ['Conv_1x1']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}

        return module_dict


class U_Net_Encoder(nn.Module):
    def __init__(self, img_ch=1, projection_size=128):
        super(U_Net_Encoder, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        hidden_size = nb_filter[4]
        projection_size = projection_size
        #nb_filter = [32,64, 128, 256, 512]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        # encoding path
        features = []
        x1 = self.Conv1(x)
        features.append(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        features.append(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        features.append(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        features.append(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        features.append(x5)

        # decoding + concat path
        x5_pool = self.Avgpool(x5)

        return x5_pool

def calc_parameters_count(model):
    """

    :param model:
    :return: The number of params in model.(M)
    """
    return np.sum(np.prod(v.size()) for v in model.parameters())/ 1e6



if __name__=="__main__":
    model = U_Net(img_ch=1, output_ch=3, normalization='softmax')
    a = torch.randn([2, 1, 128, 128])
    b = model(a)
    print(b[0,:, 2, 2])




    #summary(model, input_size=[(1, 320, 320)], batch_size=1, device="cpu")

