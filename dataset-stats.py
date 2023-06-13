import os
from collections import defaultdict

dataset_name = "lego-color-10-more-photos"


# Dictionary to store the counts
file_counts = defaultdict(int)

# Walk through the directory including subdirectories
for foldername, subfolders, filenames in os.walk(f"./datasets/{dataset_name}"):
    for filename in filenames:
        # Count the number of files in each directory
        file_counts[foldername] += 1

# Display the results
for folder, count in file_counts.items():
    print(f"{folder}: {count} files")
