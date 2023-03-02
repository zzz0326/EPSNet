# from model_erp import Saliency_Gen
import torch
import glob
import torchvision.transforms as transforms
import os
import cv2
from constants import *
from TransLib import ERP2CMP





class loader(object):
    def __init__(self):
        #reading data list
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToValInput, '*'))]
        self.size = len(self.list_img)
        self.temp = 0

    def get_batch(self):

        img = torch.zeros(self.size, 3, 160, 320)
        # sal_map = torch.zeros(self.size, 1, 160, 320)
        name = []
        to_tensor = transforms.ToTensor()
        for i in range(self.size):
            full_img_path = os.path.join(pathToValInput, self.list_img[self.temp] + '.png')
            name.append(self.list_img[self.temp])
            inputimage = cv2.imread(full_img_path)
            inputimage = cv2.resize(inputimage, (320, 160))
            img[self.temp] = to_tensor(inputimage)



            self.temp += 1

        return img, name





