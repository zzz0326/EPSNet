import argparse
import os.path as osp
import time
from typing import Union

import cv2

import numpy as np

from PIL import Image

import torch

from torchvision import transforms

from equilib import Equi2Equi
from equilib import Equi2Cube
from equilib import Cube2Equi


def preprocess(
        img: Union[np.ndarray, Image.Image], is_cv2: bool = False, gray: bool = False
) -> torch.Tensor:
    """Preprocesses image"""
    if isinstance(img, np.ndarray) and is_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, Image.Image) and gray == False:
        # Sometimes images are RGBA
        img = img.convert("RGB")

    to_tensor = transforms.Compose([transforms.ToTensor()])
    img = to_tensor(img)
    if gray == False:
        assert img.shape[0] == 3, "input must be HWC"
    assert len(img.shape) == 3, "input must be dim=3"

    return img


def postprocess(
        img: torch.Tensor, to_cv2: bool = False
) -> Union[np.ndarray, Image.Image]:
    if to_cv2:
        img = np.asarray(img.to("cpu").numpy() * 255, dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        to_PIL = transforms.Compose([transforms.ToPILImage()])
        img = img.to("cpu")
        img = to_PIL(img)
        return img


def RotateERP(path: str, yaw, roll, pitch):
    """Test single image"""
    # Rotation:
    rot = {
        "roll": roll * np.pi,  #
        "pitch": pitch * np.pi,  # vertical
        "yaw": yaw * np.pi,  # horizontal
    }

    # Initialize equi2equi
    equi2equi = Equi2Equi(height=320, width=640, mode="bilinear")
    device = torch.device("cuda")

    # Open Image
    src_img = Image.open(path).resize((320, 160))
    src_img = preprocess(src_img)

    out_img = equi2equi(src=src_img, rots=rot)
    # out_img = postprocess(out_img)
    #
    # out_img.save(outPath)
    return out_img


def ERP2CMP(src, yaw, roll, pitch):
    trans = Equi2Cube(w_face=80, cube_format='list')

    rot = {
        "roll": roll * np.pi,  #
        "pitch": pitch * np.pi,  # vertical
        "yaw": yaw * np.pi,  # horizontal
    }

    out_img = trans(equi=src, rots=rot)

    return out_img


def CMP2ERP(list_CMP, outPath):
    trans = Cube2Equi(width=320, height=160, cube_format='list')
    # device = torch.device("cuda")

    out_img = trans(list_CMP)
    out_img = postprocess(out_img)

    out_img.save(outPath)


