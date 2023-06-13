import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import os
import re
from lego_colors import lego_colors_by_id
import random
from src.bounding_box import BoundingBox

# Percentage (0-1.0) of generated images that will be used for validation,
# the remaining images for training. Setting this to 0 when experimenting
# makes it easier to review the results
percent_val = 0.2

# Folder name for this dataset
# Update to make it easier to distiguish from other versions
dataset_name = "lego-color-10-more-photos"
dataset_folder = f"./datasets/{dataset_name}"

os.makedirs("./tmp", exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)

# Do not include renders right now, they don't seem to help
# Maybe they will be valuable if we can color match the renderings
# os.system(f"cp -r ../lego-rendering/renders/lego-color-common-5k-train-only/* {dataset_folder}/")

model = YOLO("lego-detect-13-7k-more-negatives3.pt")

# Each source image has parts of a single color
# The color id is in the filename
def split_single_color_images(source_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Opening {file}...")
                filepath = os.path.join(root, file)
                img = Image.open(filepath)

                prefix = file.split('.')[0]

                id = re.findall(r'\.(\d+)\.', file)[0]
                os.makedirs(f"{dataset_folder}/train/{id}", exist_ok=True)
                os.makedirs(f"{dataset_folder}/val/{id}", exist_ok=True)

                # Setup a copy of the image to draw on
                img_copy = img.copy()
                draw = ImageDraw.Draw(img_copy)

                color = lego_colors_by_id[int(id)]

                results = model(img.convert("RGB"))

                for yolo_box in results[0].cpu().boxes:
                    part_box = BoundingBox.from_yolo(yolo_box)

                    # bounding box in the original image
                    part_box.draw(draw)
                    part_box.draw_label(draw, f"{color.name} ({color.id})",
                                    text_color = 'black',
                                    swatch_color=color.hex())

                    val_or_train = 'val' if random.random() <= percent_val else 'train'
                    part_filename = f"{dataset_folder}/{val_or_train}/{id}/{prefix}-{id}-{part_box.hash}.png"
                    part = part_box.crop(img)
                    part.save(part_filename)

                img_copy.save(f"tmp/preview-{file}")

split_single_color_images("./src/images/1x1")
split_single_color_images("./src/images/1x1-cropped")
split_single_color_images("./src/images/1x1-old")

# Each image in this folder has parts with 9 different colors
# with the colors arranged into a 3x3 grid. This lets us divide the
# image into 9 cells and we can get the color id from the filename
for root, _, files in os.walk("./src/images/3x3"):
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
                os.makedirs(f"{dataset_folder}/train/{id}", exist_ok=True)
                os.makedirs(f"{dataset_folder}/val/{id}", exist_ok=True)

            # Get the width and height of the image
            width, height = img.size
            cell_width = width // 3
            cell_height = height // 3

            # Setup a copy of the image to draw on
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)

            for row in range(3):
                for col in range(3):
                    # Define the bounding box for the cell
                    cell_box = BoundingBox.from_xywh(
                        x=col * cell_width,
                        y=row * cell_height,
                        w=cell_width,
                        h=cell_height
                    )

                    # Crop the image to the bounding box
                    cell = cell_box.crop(img)
                    cell_id = ids[row*3+col]
                    cell_color = lego_colors_by_id[cell_id]

                    results = model(cell.convert("RGB"))

                    for yolo_box in results[0].cpu().boxes:
                        part_box = BoundingBox.from_yolo(yolo_box)

                        # bounding box in the original image
                        box = part_box.move(
                            cell_box.x,
                            cell_box.y
                            )
                        box.draw(draw)
                        box.draw_label(draw, f"{cell_color.name} ({cell_color.id})",
                                       text_color = 'black',
                                       swatch_color=cell_color.hex())

                        val_or_train = 'val' if random.random() <= percent_val else 'train'
                        part_filename = f"{dataset_folder}/{val_or_train}/{cell_id}/{prefix}-{cell_id}-{part_box.hash}.png"
                        part = part_box.crop(cell)
                        part.save(part_filename)

            img_copy.save(f"tmp/preview-{file}")

os.chdir('datasets')
# os.system(f'zip -r {dataset_name}.zip {dataset_name}')
os.chdir('..')

print('')
print(f"Created dataset at {dataset_folder}")
# print(f"Zipped to {dataset_name}.zip")
print("See tmp/preview-* to see how the source images were parsed")
print('')
print("Done")
