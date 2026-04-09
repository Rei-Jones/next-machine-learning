import os

from PIL import Image

import numpy as np

from ultralytics import YOLO

import sys

id = int(sys.argv[1])
model = YOLO("yolo26n.yaml")

def convert_rgba_to_rgb(img_dir):
    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            path = os.path.join(img_dir, f)
            img = Image.open(path).convert("RGB")
            img.save(path)

convert_rgba_to_rgb("dataset/images/train")
convert_rgba_to_rgb("dataset/images/val")

search_space = {
    "lr0": (1e-5, 1e-2),
    "lrf": (0.01, 1.0),
    "momentum": (0.7, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (1.0, 20.0),
    "cls": (0.1, 4.0),
    "dfl": (0.4, 12.0),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.95),
    "shear": (0.0, 10.0),
    "perspective": 	(0.0, 0.001),
    "flipud": (0.0, 0.0),
    "fliplr": (0.0, 0.0),
    "mosaic": (0.0, 0.0),
    "copy_paste": (0.0, 0.0),
    "close_mosaic": (0.0, 0.0),
    "degrees": (0.0, 45.0),
    "bgr": (0.0, 0.0),
    "mixup": (0.0, 0.0),
    "hsv_v": (0.0, 0.0),
    "hsv_s": (0.0, 0.0),
    "hsv_h": (0.0, 0.0)
}

print(search_space)

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="data.yaml",
    epochs=30,
    imgsz=416,
    iterations=50,
    optimizer="AdamW",
    space=search_space,
    plots=True,
    save=True,
    val=True,
    project="./runs/tune_results",
    name=f"results_{id}"
)
