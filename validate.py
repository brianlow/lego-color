import os
import sys
import zipfile

from ultralytics import YOLO
from pathlib import Path
from lego_colors import lego_colors_by_id

dataset_name = "lego-color-common-5k-dataset-trans-real"
model = YOLO("color-03-common-5k-trans-real2.pt")

data_dir = './datasets'
data_dir = str(Path(data_dir).resolve())
dataset_dir = os.path.join(data_dir, dataset_name)
per_class_datasets_dir = os.path.join(dataset_dir, "per-class-datasets")

os.makedirs(per_class_datasets_dir, exist_ok=True)

class_dirs = [f.path for f in os.scandir(f"{dataset_dir}/val") if f.is_dir()]
class_names = [os.path.basename(class_dir) for class_dir in class_dirs]

for class_dir in class_dirs:
    class_name = os.path.basename(class_dir)
    color = lego_colors_by_id[int(class_name)]

    class_dataset_dir = os.path.join(per_class_datasets_dir, f"dataset-{class_name}")
    dest_dir = os.path.join(class_dataset_dir, "val", class_name)

    print(class_dataset_dir)
    print(f"{class_dir} -> {dest_dir}")

    if not os.path.exists(class_dataset_dir):
        os.makedirs(os.path.join(class_dataset_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(class_dataset_dir, "val"), exist_ok=True)
        os.symlink(class_dir, dest_dir)
        for other_class_name in class_names:
            os.makedirs(os.path.join(class_dataset_dir, "train", other_class_name), exist_ok=True)

    model.val(data=class_dataset_dir, name=f"val-{color.id}-{color.name}", hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)

    print(f"{color.id} - {color.name} -> {model.metrics.top1:.1%} accuracy (top 1)")




# Validate
# model.val(data=f"{dataset_dir}, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)

# print(f"Accuracy Top 1: {model.metrics.top1:.1%}")
