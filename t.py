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

model = YOLO("color-03-common-nano.pt")



img = Image.open("../lego-camera/tmp/last-classify-transform.png")
color_results = model(img.convert("RGB"))

color_result = color_results[0].cpu()
print(color_result.names)
print(color_result.probs)

class_dict = color_result.names
pred_tensor = color_result.probs

# Get the top 3 indices and values
topk_values, topk_indices = torch.topk(
    pred_tensor, k=3)

# Get the corresponding class labels from the class dictionary
topk_classes = [class_dict[i.item()]
                for i in topk_indices]

# Print the top 3 classes and their corresponding probabilities
for i in range(len(topk_classes)):
    predicted = lego_colors_by_id[int(topk_classes[i])]
    confidence = topk_values[i]
    print(
        f"    {confidence * 100:.0f}%: ({predicted.id}) {predicted.name} ")
