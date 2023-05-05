import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import os
import re
import hashlib
from lego_colors import lego_colors_by_id
import random

# Percentage (0-1.0) of generated images that will be used for validation,
# the remaining images for training. Setting this to 0 when experimenting
# makes it easier to review the results
percent_val = 0.2

os.makedirs("./tmp", exist_ok=True)
os.makedirs("./data/dataset", exist_ok=True)

model = YOLO("detect-10-4k-real-and-renders-nano-1024-image-size2.pt")

all_ids = set([])

for root, _, files in os.walk("./src"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Opening {file}...")
            filepath = os.path.join(root, file)
            img = Image.open(filepath)

            prefix = file.split('.')[0]

            id_strings = re.findall(r'\.(.*?)\.', file)
            ids = [int(n) for n in id_strings[0].split('-')]
            print("ids: ", ids)
            for id in ids:
                os.makedirs(f"./data/dataset/train/{id}", exist_ok=True)
                os.makedirs(f"./data/dataset/val/{id}", exist_ok=True)
            all_ids.update(ids)

            # Get the width and height of the image
            width, height = img.size
            cell_width = width // 3
            cell_height = height // 3

            for i in range(3):
                for j in range(3):
                    # Define the bounding box for the cell
                    left = j * cell_width
                    top = i * cell_height
                    right = (j + 1) * cell_width
                    bottom = (i + 1) * cell_height

                    # Crop the image to the bounding box
                    cell = img.crop((left, top, right, bottom))
                    cell_id = ids[i*3+j]

                    results = model(cell.convert("RGB"))
                    print("cell size: ", cell.size)

                    for box in results[0].cpu().boxes:
                        bounding_box_bytes = str(box.xyxy[0]).encode('utf-8')
                        hash = hashlib.sha256(bounding_box_bytes).hexdigest()[:6]

                        val_or_train = 'val' if random.random() <= percent_val else 'train'
                        part_filename = f"./data/dataset/{val_or_train}/{cell_id}/{prefix}-{cell_id}-{hash}.png"
                        print(part_filename)
                        part = cell.crop(box.xyxy[0].int().numpy())
                        part.save(part_filename)


print("done.")
