import os
import sys
import zipfile
import torch

from ultralytics import YOLO
from pathlib import Path
from lego_colors import lego_colors_by_id

dataset_name = "lego-color-11-yellows-and-blues"
model = YOLO("lego-color-common-5k-dataset-4-baseline-plus-renders.pt")

data_dir = './datasets'
data_dir = str(Path(data_dir).resolve())
dataset_dir = os.path.join(data_dir, dataset_name)

total = 0
correct = 0
accuracies = []

class_dirs = [f.path for f in os.scandir(f"{dataset_dir}/val") if f.is_dir()]
for class_dir in class_dirs:
    class_name = os.path.basename(class_dir)
    color = lego_colors_by_id[int(class_name)]
    print(f"Validating {color.id} - {color.name}...")
    class_total = 0
    class_correct = 0

    for root, _, files in os.walk(class_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Opening {file}...")
                filepath = os.path.join(root, file)

                results = model(filepath)

                result = results[0].cpu()
                class_dict = result.names
                pred_tensor = result.probs
                topk_values, topk_indices = torch.topk(pred_tensor, k=1)
                topk_classes = [class_dict[i.item()]for i in topk_indices]
                predicted = lego_colors_by_id[int(topk_classes[0])]

                class_total += 1
                total += 1
                if predicted.id == color.id:
                    class_correct += 1
                    correct += 1
                print(f"Predicted: {predicted.id} - {predicted.name} -> {predicted.id == color.id}")


    accuracy = class_correct/class_total

    accuracies.append((color, accuracy, class_total))


print("")
print("Top1 Accuracy by Color")
print("-----------")
print("")

# sort accuracies by accuracy descending
accuracies.sort(key=lambda x: x[1], reverse=True)

for color, accuracy, class_total in accuracies:
    print(f"{color.id:4d} - {color.name:<20s}:  {accuracy:6.1%} ({class_total} images)")

print("")
print(f"Overall accuracy {correct/total:.1%}")
