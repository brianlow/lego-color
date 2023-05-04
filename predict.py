from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import torch
import os


os.makedirs("./tmp", exist_ok=True)

# Open the image
img = Image.open('./src/blues.jpeg')

# Get the width and height of the image
width, height = img.size

# Calculate the size of each grid cell
cell_width = width // 3
cell_height = height // 3

# Create a list to store the cropped images
cells = []

# Loop through each cell in the grid
for i in range(3):
    for j in range(3):
        # Define the bounding box for the cell
        left = j * cell_width
        top = i * cell_height
        right = (j + 1) * cell_width
        bottom = (i + 1) * cell_height

        # Crop the image to the bounding box
        cell = img.crop((left, top, right, bottom))

        # Add the cell to the list of cells
        cells.append(cell)

# # Create a new figure
# fig, axs = plt.subplots(3, 3, figsize=(8, 8))
#
# # Loop through each cell in the grid
# for i in range(3):
#     for j in range(3):
#         # Plot the cell in the corresponding subplot
#         axs[i, j].imshow(cells[i*3+j])
#         axs[i, j].axis('off')
#
#
#
# fig.savefig(fname="./tmp/sliced.png")




model = YOLO("detect-10-4k-real-and-renders-nano-1024-image-size2.pt")

# Create a new figure
# fig, axs = plt.subplots(3, 3, figsize=(8, 8))

# # Loop through each cell in the grid
# for i in range(3):
#     for j in range(3):
original = cells[7]
results = model(original.convert("RGB"))
boxes = results[0].cpu().boxes

scratch = original.copy()
draw = ImageDraw.Draw(scratch)

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    print(box.xyxy[0])
    print(x1, y1, x2, y2)

    draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=2)

scratch.save("./tmp/first-color.png")



box = boxes[0]
x1, y1, x2, y2 = box.xyxy[0]

image = original.crop((int(x1), int(y1), int(x2), int(y2)))
image.save("./tmp/first-lego.png")


img_array = np.array(image)

img_array_2d = img_array.reshape(-1, 3)

# Perform k-means clustering with 5 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(img_array_2d)

dominant_colors = kmeans.cluster_centers_.astype(int)
cluster_sizes = np.bincount(kmeans.labels_)
print(f"dominant_colors: #{dominant_colors}")
print(f"cluster_sizes: #{cluster_sizes}")

from lego_colors import lego_colors_by_id, lego_colors

# def compare(id1, id2):
#     color1 = lego_colors_by_id[id1]
#     color2 = lego_colors_by_id[id2]
#     distance = color1.distance(color2)
#     print(f"{color1.name} vs {color2.name} -> {distance}")
#
# compare(1, 1)
# compare(1, 73)
# compare(1, 272)
# compare(1, 0)

print("---------")

print(lego_colors_by_id[1].distance(dominant_colors[0]))

print("----------")

def my_distance(color):
    return color.distance(dominant_color)

dominant_color = dominant_colors[0]
closest_colors = sorted(lego_colors, key=my_distance)
for closest_colors in closest_colors[:15]:
    print(closest_colors)



# import numpy as np
# from skimage import io
# from skimage.color import rgb2lab, deltaE_cie76
#
# rgb = io.imread('https://i.stack.imgur.com/npnrv.png')
# lab = rgb2lab(rgb)
#
# green = [0, 160, 0]
# magenta = [120, 0, 140]
#
# threshold_green = 15
# threshold_magenta = 20
#
# green_3d = np.uint8(np.asarray([[green]]))
# magenta_3d = np.uint8(np.asarray([[magenta]]))
#
# dE_green = deltaE_cie76(rgb2lab(green_3d), lab)
# dE_magenta = deltaE_cie76(rgb2lab(magenta_3d), lab)
#
# rgb[dE_green < threshold_green] = green_3d
# rgb[dE_magenta < threshold_magenta] = magenta_3d
# Image.fromarray(rgb).show()

print("done.")
