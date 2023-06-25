import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import os
import re
from lego_colors import lego_colors_by_id
import torch
import random
from src.bounding_box import BoundingBox

detection_model = YOLO("lego-detect-13-7k-more-negatives3.pt")
model = YOLO("lego-color-11-yellows-and-blues-nano2.pt")

font_path = os.path.expanduser('~/Library/Fonts/Arial.ttf')
font = ImageFont.truetype(font_path, size=24)

def predict_single_color_images(source_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Opening {file}...")
                filepath = os.path.join(root, file)
                img = Image.open(filepath)

                prefix = file.split('.')[0]

                # Setup copy of the image to draw on
                img_copy = img.copy()
                draw = ImageDraw.Draw(img_copy)

                detection_results = detection_model(img.convert("RGB"))

                for yolo_box in detection_results[0].cpu().boxes:
                    part_box = BoundingBox.from_yolo(yolo_box)
                    part = part_box.crop(img)

                    color_results = model(part.convert("RGB"))

                    color_result = color_results[0].cpu()
                    print(color_result.names)
                    print(color_result.probs.data)

                    class_dict = color_result.names
                    pred_tensor = color_result.probs.data

                    # Get the top 3 indices and values
                    topk_values, topk_indices = torch.topk(
                        pred_tensor, k=3)

                    # Get the corresponding class labels from the class dictionary
                    topk_classes = [class_dict[i.item()]
                                    for i in topk_indices]

                    # bounding box in the original image
                    part_box.draw(draw)

                    predicted = lego_colors_by_id[int(topk_classes[0])]
                    confidence = topk_values[0]
                    part_box.draw_label(draw, f"{confidence * 100:.0f}%: {predicted.name} ({predicted.id})",
                                    text_color = 'black',
                                    swatch_color=predicted.hex())

                img_copy.save(f"tmp/predict-{file}")

def predict_3x3_color_images(source_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Opening {file}...")
                filepath = os.path.join(root, file)
                img = Image.open(filepath)

                prefix = file.split('.')[0]

                id_strings = re.findall(r'\.(.*?)\.', file)
                ids = [int(n) for n in id_strings[0].split('-')]
                print("ids: ", ids)

                # Get the width and height of the image
                # We will divide it into 9 cells (3x3)
                # and perform object detection on each cell
                width, height = img.size
                cell_width = width // 3
                cell_height = height // 3

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
                        cell_box.draw(draw)

                        # Crop the image to the bounding box
                        cell = cell_box.crop(img)
                        cell_id = ids[(row*3)+col]

                        detection_results = detection_model(cell.convert("RGB"))

                        for yolo_box in detection_results[0].cpu().boxes:
                            part_box = BoundingBox.from_yolo(yolo_box)
                            part = part_box.crop(cell)

                            color_results = model(part.convert("RGB"))

                            color_result = color_results[0].cpu()
                            print(color_result.names)
                            print(color_result.probs.data)

                            class_dict = color_result.names
                            pred_tensor = color_result.probs.data

                            # Get the top 3 indices and values
                            topk_values, topk_indices = torch.topk(
                                pred_tensor, k=3)

                            # Get the corresponding class labels from the class dictionary
                            topk_classes = [class_dict[i.item()]
                                            for i in topk_indices]

                            actual = lego_colors_by_id[cell_id]
                            print(f"({actual.id}) {actual.name} ->")
                            # Print the top 3 classes and their corresponding probabilities
                            for i in range(len(topk_classes)):
                                predicted = lego_colors_by_id[int(topk_classes[i])]
                                confidence = topk_values[i]
                                print(
                                    f"    {confidence * 100:.0f}%: ({predicted.id}) {predicted.name} ")

                            # bounding box in the original image
                            box = part_box.move(
                                cell_box.x,
                                cell_box.y
                                )
                            box.draw(draw)

                            predicted = lego_colors_by_id[int(topk_classes[0])]
                            confidence = topk_values[0]
                            correct = predicted == actual
                            box.draw_label(draw, f"{confidence * 100:.0f}%: {predicted.name} ({predicted.id})",
                                        text_color = 'black' if correct else 'red',
                                        swatch_color=predicted.hex())

                        img_copy.save(f"tmp/predict-{file}")


predict_single_color_images("./src/images/1x1-yellows-and-blues")
# predict_3x3_color_images("./src/images/3x3")

print("See tmp/predict-* to predictions for source images")
print("Done")
