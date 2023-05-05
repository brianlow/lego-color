import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import os
import re
from lego_colors import lego_colors_by_id
import torch
import random

detection_model = YOLO("detect-10-4k-real-and-renders-nano-1024-image-size2.pt")
model = YOLO("color-02-tiny-paperspace3.pt")

font_path = os.path.expanduser('~/Library/Fonts/Arial.ttf')
font = ImageFont.truetype(font_path, size=24)

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

            draw = ImageDraw.Draw(img)

            for row in range(3):
                for col in range(3):
                    # Define the bounding box for the cell
                    cell_left = col * cell_width
                    cell_top = row * cell_height
                    cell_right = (col + 1) * cell_width
                    cell_bottom = (row + 1) * cell_height

                    # Crop the image to the bounding box
                    cell = img.crop((cell_left, cell_top, cell_right, cell_bottom))
                    cell_id = ids[(row*3)+col]

                    detection_results = detection_model(cell.convert("RGB"))

                    for box in detection_results[0].cpu().boxes:
                        part = cell.crop(box.xyxy[0].int().numpy())

                        color_results = model(part.convert("RGB"))

                        color_result = color_results[0].cpu()
                        print(color_result.names)
                        print(color_result.probs)

                        class_dict = color_result.names
                        pred_tensor = color_result.probs

                        # Get the top 3 indices and values
                        topk_values, topk_indices = torch.topk(pred_tensor, k=3)

                        # Get the corresponding class labels from the class dictionary
                        topk_classes = [class_dict[i.item()] for i in topk_indices]

                        actual = lego_colors_by_id[cell_id]
                        print(f"({actual.id}) {actual.name} ->")
                        # Print the top 3 classes and their corresponding probabilities
                        for i in range(len(topk_classes)):
                            predicted = lego_colors_by_id[int(topk_classes[i])]
                            confidence = topk_values[i]
                            print(f"    {confidence * 100:.0f}%: ({predicted.id}) {predicted.name} ")

                        # bounding box in the original image
                        x1 = cell_left + box.xyxy[0][0].int()
                        y1 = cell_top + box.xyxy[0][1].int()
                        x2 = cell_left + box.xyxy[0][2].int()
                        y2 = cell_top + box.xyxy[0][3].int()
                        draw.rectangle(((x1, y1), (x2, y2)), outline='white', width=2)

                        predicted = lego_colors_by_id[int(topk_classes[0])]
                        confidence = topk_values[0]
                        correct = predicted == actual
                        draw.rectangle(((x1, y2+10), (x1+25, y2+10+25)), fill=f"#{predicted.hex()}")
                        draw.text((x1+25+10, y2+10), f"{confidence * 100:.0f}%: {predicted.name} ({predicted.id})", fill='black' if correct else 'red', font=font)

                    img.save(f"tmp/{file}")




print("done.")
