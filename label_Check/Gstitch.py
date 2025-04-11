import cv2
import os
import numpy as np
import random
import sys
from collections import defaultdict
from main import extract_labels
from main import visualize_labels
from tqdm import tqdm

def pad_image(image, pad):
    image[:pad,:,:]  = 0
    image[-pad:,:,:] = 0
    image[:,:pad,:]  = 0
    image[:,-pad:,:] = 0
    return image

if len(sys.argv) <= 2:   #w/o labeling
    flag = False
    PATCH_SIZE = 64
elif sys.argv[2] == 'l':  #w labeling
    flag = True
    PATCH_SIZE = 64
else:
    raise NotImplementedError

IMG_DATA_ROOT = ''
LABEL_DATA_ROOT = 'labels/' 
SOURCE = 'naip/' if sys.argv[1].lower() == 'n' else 'sentinel2/'

data_folder = IMG_DATA_ROOT + SOURCE
tdirectories = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
random_dir = random.choice(tdirectories)
tsub_directory = os.path.join(data_folder, random_dir, 'tci')
image_files = sorted(os.listdir(tsub_directory))

# Group images by "xxxxx" prefix and store yyyyy-based locations
grouped_images = defaultdict(list)
for img in image_files:
    if '_' in img and img.endswith('.png'):
        prefix, yyyyy = img.rsplit('_', 1)
        yyyyy = yyyyy.split('.')[0]  # Remove .png
        if yyyyy.isdigit():
            row_idx = int(yyyyy)  # We can directly use yyyyy as row index
            grouped_images[prefix].append((img, row_idx))

print('xxxxx:', list(grouped_images.keys()))

num_columns = int(max(list(grouped_images.keys()))) - int(min(list(grouped_images.keys()))) + 1 #len(grouped_images)
num_rows = max(len(images) for images in grouped_images.values())
BIG_IMAGE_SIZE = PATCH_SIZE * num_rows  # Number of rows is the max number of patches per prefix
big_image = np.zeros((BIG_IMAGE_SIZE, PATCH_SIZE * num_columns, 3), dtype=np.uint8)

# Fill columns using yyyyy for row placement
col_idx = 0
prev_prefix = -1
for prefix, images in tqdm(grouped_images.items()):
    if int(prefix) - prev_prefix != 1 and prev_prefix != -1:
        jump_num = int(prefix) - prev_prefix - 1
        big_image[:, col_idx * PATCH_SIZE:(col_idx + jump_num) * PATCH_SIZE] = np.zeros((BIG_IMAGE_SIZE, jump_num*PATCH_SIZE, 3), dtype=np.uint8)
        col_idx += jump_num

    placed_rows = set()  # Keep track of filled row positions
    images = sorted(images, key=lambda x: x[1])  # Sort by yyyyy value for correct row order
    for image_file, row_idx in images:
        row_idx = row_idx % num_rows
        placed_rows.add(row_idx)
        image_path = os.path.join(tsub_directory, image_file)

        if flag:                            #w labeling
            image, json_file, tlbr = extract_labels(image=image_path, labels=LABEL_DATA_ROOT, src=SOURCE)
            image = visualize_labels(image=image, json_file=json_file, src=SOURCE, tlbr=tlbr)
        else:
            image = cv2.imread(image_path)  #w/o labeling

        image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
        image = pad_image(image=image, pad=1)
        big_image[row_idx * PATCH_SIZE:(row_idx + 1) * PATCH_SIZE, col_idx * PATCH_SIZE:(col_idx + 1) * PATCH_SIZE] = image
    col_idx += 1
    prev_prefix = int(prefix)

while col_idx < num_columns:
    for row_idx in range(num_rows):
        big_image[row_idx * PATCH_SIZE:(row_idx + 1) * PATCH_SIZE, col_idx * PATCH_SIZE:(col_idx + 1) * PATCH_SIZE] = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    col_idx += 1

output_path = f"/home/ghahramani/GeoPixel/label_Check/composite/out.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, big_image)

print(f"Final image shape: ({int(big_image.shape[0]/PATCH_SIZE)}x{PATCH_SIZE} , {int(big_image.shape[1]/PATCH_SIZE)}x{PATCH_SIZE})")
print(f"Image folder: {tsub_directory}")
# print(f"Composite image saved: {output_path}") 
