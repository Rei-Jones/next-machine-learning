import numpy as np

from ultralytics import YOLO

import sys

id = int(sys.argv[1])
model = YOLO("yolo26n.pt")

def get_range(start, interval, id):
    range = ((start+(id*interval)), (start+(id*interval)+interval))
    return range

search_space = {
    "lr0": get_range(1e-5, 0.009999, id),
    "lrf": get_range(0.01, 0.099, id),
    "weight_decay": get_range(0.0, 1e-4, id),
    "warmup_epochs": get_range(0, 0.5, id),
    "warmup_momentum": get_range(0, 0.095, id),
    "box": get_range(0.02, 0.018, id),
    "cls": get_range(0.2, 0.38, id),
    "dfl": get_range(0.4, 0.56, id),
    "translate": get_range(0.0, 0.09, id),
    "scale": get_range(0.0, 0.09, id),
    "shear": get_range(0.0, 1.0, id),
    "perspective": get_range(0.0, 0.0001, id),
    "flipud": (0.0, 0.0),
    "fliplr": (0.0, 0.0),
    "mosaic": get_range(0.0, 0.1, id),
    "copy_paste": (0.0, 0.0),
    "close_mosaic": get_range(0, 1, id),
    "degrees": get_range(0, 4.5, id),
    "bgr": (0.0, 0.0),
    "mixup": (0.0, 0.0),
    "hsv_v": (0.0, 0.0),
    "hsv_s": (0.0, 0.0),
    "hsv_h": (0.0, 0.0)
}

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="data.yaml",
    epochs=100,
    iterations=30,
    optimizer="AdamW",
    space=search_space,
    plots=True,
    save=True,
    val=True,
    runs_dir="./runs",
    project="tune_results",
    name=f"results_{id}"
)