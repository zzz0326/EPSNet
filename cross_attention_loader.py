import glob
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import dataset
from TransLib import ERP2CMP
from PIL import Image
from TransLib import preprocess

from constants import *


class LoadData(dataset.Dataset):

    def __init__(self):
        super(LoadData, self).__init__()

        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToEncoderTrain, '*'))]
        self.size = len(self.list_img)
        pass

    def __getitem__(self, index):


        transforms_list = [
            transforms.Resize(90),
            transforms.RandomCrop(80),
        ]
        trans = transforms.Compose(transforms_list)

        # shuffle
        x = torch.arange(0, 6)
        x = torch.randperm(x.size(0))

        curr_file = self.list_img[index]
        ERP = Image.open(pathToEncoderTrain + curr_file + '.jpg').resize((320, 160))
        ERP = preprocess(ERP)


        cmp_list = ERP2CMP(ERP, 0, 0, 0)
        temp_cmp = []
        for i in range(6):
            temp_cmp.append(trans(cmp_list[i]))

        x = x.clone().detach().to(torch.float32)


        return ERP, temp_cmp[int(x[0])], temp_cmp[int(x[1])], temp_cmp[int(x[2])], temp_cmp[int(x[3])], temp_cmp[int(x[4])], temp_cmp[int(x[5])], x

    def __len__(self):
        return self.size


