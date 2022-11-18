import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadImages

source = "data/images"
imgsz= 128
stride= 32
save_path = "C:/Users/kikuchilab/PycharmProjects/yolo5/test/2H6A9795.jpg"

dataset = LoadImages(source, img_size=imgsz, stride=stride)
data_name = "saisyo"
for path, img, im0s, vid_cap, s in dataset:
    if data_name == "saisyo":
        data_name = Path(path).stem[:8]
    if Path(path).stem[:8] != data_name:
        break
    """
    print(type(im0s))
    #print(im0s)
    """
    print(type(path))
    print(path)
    print(im0s.shape)


    cv2.imshow('color', im0s)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    cv2.imwrite(save_path, im0s)
    """
    #print(type(img))
