import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from constants import *

class DataLoader(object):

    def __init__(self, batch_size=5):
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToDecoderInput, '*'))]
        self.batch_size = batch_size
        self.size = len(self.list_img)
        self.cursor = 0
        self.num_batches = self.size / batch_size

    def get_4000batch(self):
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)

        img = torch.zeros(self.batch_size, 3, 160, 320)
        sal_map = torch.zeros(self.batch_size, 1, 160, 320)
        fix_map = torch.zeros(self.batch_size, 1, 160, 320)

        to_tensor = transforms.ToTensor()  # Transforms 0-255 numbers to 0 - 1.0.

        for idx in range(self.batch_size):
            curr_file = self.list_img[self.cursor]
            video_id = curr_file.split('o')[1].split('_')[0]
            if int(video_id) < 10:
                video_id = '00' + video_id
            elif int(video_id) < 100:
                video_id = '0' + video_id

            frame_id = curr_file.split('_')[1]

            full_img_path = os.path.join(pathToDecoderInput, curr_file + '.png')
            full_map_path = os.path.join(pathToDecoderSaliencyMap, video_id + 'video_' + frame_id + '.png')
            full_fix_path = os.path.join(pathToDecoderFixationMap, video_id + 'video_' + frame_id + '.png')
            self.cursor += 1
            inputimage = cv2.imread(full_img_path)  # (192,256,3)
            inputimage = cv2.resize(inputimage, (320, 160))
            img[idx] = to_tensor(inputimage)

            saliencyimage = cv2.imread(full_map_path, 0)
            saliencyimage = np.expand_dims(saliencyimage, axis=2)
            saliencyimage = cv2.resize(saliencyimage, (320, 160))
            sal_map[idx] = to_tensor(saliencyimage)

            FIximage = cv2.imread(full_fix_path, 0)
            Fiximage = np.expand_dims(FIximage, axis=2)
            fix_map[idx] = to_tensor(Fiximage)

        return (img, sal_map, fix_map)





