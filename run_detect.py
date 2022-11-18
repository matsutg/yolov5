import detect
from detect import run
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# from utils.get_feature_reaction import GetFeature  # 反応分布取得用
import temp_x

check_requirements(exclude=('tensorboard', 'thop'))

run(weights="runs/train/exp/weights/best.pt",
    source="data/test/images/train_41_01826.png",
    imgsz=[128, 128],
    save_txt=True,  # save results to *.txt
    save_conf=True,  # save confidences in --save-txt labels
    nosave=True,
    save_conv=False,
    calc_dist=False,
    deform=False,
    shear=False,  # 欠損
    distort=False,  # 歪み
    search_min=False,
    plot_conv=False
    )

"""
load_path = "C:/Users/kikuchilab/PycharmProjects/yolo5/Imagehash/220211/epoch100/AP7A7553/crops"
labels_list = os.listdir(load_path)
print(labels_list)
for i, label in enumerate(labels_list):
    folder_path = os.path.join(load_path, label)
    run(weights="runs/train/exp/weights/best.pt",
        source= folder_path,
        imgsz=[128, 128],
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        nosave=True,

        calc_dist=True,
        deform=False,
        search_min=True
        )
"""
"""
run(weights="runs/train/exp/weights/best.pt",
    source="C:/Users/kikuchilab/PycharmProjects/yolo5/Imagehash/220211/epoch100/2H6A6850/crops/G/1431_2H6A6850_0.29946810007095337.jpg",
    imgsz=[128, 128],
    save_txt=True,  # save results to *.txt
    save_conf=True,  # save confidences in --save-txt labels
    nosave=True,
    calc_dist=True,
    deform=False,
    search_min=True
    )  # python detect.py --source C:/Users/kikuchilab/PycharmProjects/yolo5/Imagehash/220211/epoch100/2H6A9795/crops/f --weights runs/train/exp/weights/best.pt --img 128 --nosave
"""
