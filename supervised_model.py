import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn.functional import interpolate




def conv2d(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)


def deconv2d(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)


def relu(inplace=True):  # Change to True?
    return nn.ReLU(inplace)


def maxpool2d():
    return nn.MaxPool2d(2)


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_deconv_layers(cfg):
    layers = []
    in_channels = 512
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 'U', 512, 'U', 256, 'U', 128, 'U', 64, 1]
}


def encoder():
    return make_conv_layers(cfg['E'])


def decoder():
    return make_deconv_layers(cfg['D'])


class Model_rethink(nn.Module):
    def __init__(self):
        super(Model_rethink, self).__init__()
        self.erp_encoder = encoder()


        decoder_salgan = [
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            Sigmoid(),
        ]
        self.saliency_decoder = torch.nn.Sequential(*decoder_salgan)

        self.erp_encoder.requires_grad = False

    def forward(self, erp):
        erp = self.erp_encoder(erp)
        erp = self.saliency_decoder(erp)

        return erp


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = 'bilinear'

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners = True)
        return x



