import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

width = 2048
height = 1024

# batch = 1时的方法

def KL_div(x, y):

    kld = 0
    for i in range(x.shape[0]):
        output = x[i, ]
        target = y[i, ]
        # w = target.mean()
        # print(w)
        output = output / output.sum()
        target = target / target.sum()
        a = target * (target / (output + 1e-50) + 1e-50).log()
        b = output * (output / (target + 1e-50) + 1e-50).log()
        kld += ((a.sum() + b.sum()) / 2)
    return kld/x.shape[0]


def CC(x, y):

    cc = 0
    for i in range(x.shape[0]):
        output = x[i, ]
        target = y[i, ]
        output = (output - output.mean()) / output.std()
        target = (target - target.mean()) / target.std()

        num = (output - output.mean()) * (target - target.mean())
        out_squre = torch.square(output - output.mean())
        tar_squre = torch.square(target - target.mean())

        cc += (num.sum() / (torch.sqrt(out_squre.sum() * tar_squre.sum())))
    return cc/x.shape[0]


def NSS(x, y):

    nss = 0
    for i in range(x.shape[0]):
        output = x[i, ]
        fixationMap = y[i, ]
        output = (output - output.mean()) / output.std()
        Sal = output * fixationMap
        # print(fixationMap.sum())
        nss += (Sal.sum() / fixationMap.sum())
    return nss/x.shape[0]





class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, map, target, fix):
        kld = KL_div(map, target)
        cc = CC(map, target)
        nss = NSS(map, fix)
        return kld, cc, nss





