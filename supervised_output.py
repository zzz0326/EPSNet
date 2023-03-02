from supervised_model import Model_rethink
import torch

import cv2
from Val_loader import loader

from constants import *



model = Model_rethink().cpu()
weight = torch.load(
    "saliency_model.pkl", map_location='cpu')
model.load_state_dict(weight, strict=True)

data = loader()
input, name = data.get_batch()

out = model(input).detach().numpy()

for i in range(out.shape[0]):
    temp = out[i,]
    temp = temp.transpose(1, 2, 0)

    temp = temp * 255
    cv2.imwrite('./generator_output/' + name[i] + '.png', temp)

    binname = './generator_output/' + name[i] + '.bin'
    temp.tofile(binname)

