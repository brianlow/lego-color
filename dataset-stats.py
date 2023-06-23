import os
from collections import defaultdict

dataset_name = "lego-color-11-yellows-and-blues"

dataset_path = f"./datasets/{dataset_name}"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

# Dictionary to store the counts
file_counts = defaultdict(lambda: (0,0))

classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
classes += [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
classes = sorted(set([int(c) for c in classes]))

train_total = 0
val_total = 0
grand_total = 0

print("Class:  # Train   # Val   Total")
print("------  -------   -----   -----")
for class_name in classes:
    class_train_path = os.path.join(train_path, str(class_name))
    class_val_path = os.path.join(val_path, str(class_name))

    # Count the number of images in each folder
    train_count = len(os.listdir(class_train_path))
    val_count = len(os.listdir(class_val_path))

    train_total += train_count
    val_total += val_count
    grand_total += train_count + val_count

    # Print the results
    print(f"{class_name:<5}:     {train_count:>4}    {val_count:>4}   {(train_count + val_count):>5}")

print("------  -------   -----   -----")
print(f"Total      {train_total:>4}    {val_total:>4}   {(grand_total):>5}")
