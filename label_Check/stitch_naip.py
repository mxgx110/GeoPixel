import cv2
import os
import numpy as np
import sys
from collections import defaultdict

def pad_image(image, pad):
    image[:pad,:,:]  = 0
    image[-pad:,:,:] = 0
    image[:,:pad,:]  = 0
    image[:,-pad:,:] = 0
    return image

PATCH_SIZE = 512
GRID_SIZE = 16  # Each tile is 16x16 patches (8kx8k)

# Data path
IMG_DATA_ROOT = ''
SOURCE = 'naip/' if sys.argv[1].lower() == 'n' else 'sentinel2/'
data_folder = IMG_DATA_ROOT + SOURCE

# Choose a random directory
tdirectories = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
random_dir = np.random.choice(tdirectories)
tsub_directory = os.path.join(data_folder, random_dir, 'tci')
image_files = sorted(os.listdir(tsub_directory))

# Group images by xxxxx and yyyyy
tiles = defaultdict(lambda: defaultdict(lambda: np.zeros((PATCH_SIZE * GRID_SIZE, PATCH_SIZE * GRID_SIZE, 3), dtype=np.uint8)))

for img in image_files:
    if '_' in img and img.endswith('.png'):
        xxxxx, yyyyy = img.rsplit('_', 1)
        yyyyy = yyyyy.split('.')[0]
        if xxxxx.isdigit() and yyyyy.isdigit():
            xxxxx, yyyyy = int(xxxxx), int(yyyyy)
            P, Q = xxxxx // GRID_SIZE, yyyyy // GRID_SIZE
            col_idx, row_idx = xxxxx % GRID_SIZE, yyyyy % GRID_SIZE
            image_path = os.path.join(tsub_directory, img)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
            image = pad_image(image=image, pad=1)
            tiles[P][Q][row_idx * PATCH_SIZE:(row_idx + 1) * PATCH_SIZE, col_idx * PATCH_SIZE:(col_idx + 1) * PATCH_SIZE] = image

output_dir = "/home/ghahramani/GeoPixel/label_Check/composite/"
os.makedirs(output_dir, exist_ok=True)
for P in tiles:
    for Q in tiles[P]:
        output_path = os.path.join(output_dir, f"{P}_{Q}.png")
        cv2.imwrite(output_path, tiles[P][Q])
        print(f"Saved tile {P}_{Q}.png at {output_path}")
