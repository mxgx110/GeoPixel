import numpy as np   
import cv2
import json
import os
import random
import subprocess
from PIL import Image 
import sys

def extract_labels_naip(tile, labels):
    tile_name = (tile.split('/')[-1].split('.')[0])[10:19]
    for lb_dir in os.listdir(labels):
        if lb_dir.startswith('static'):
            st_folder = lb_dir + '/'
            for dir in os.listdir(labels + st_folder):
                if dir == tile_name:
                    json_file = labels + st_folder + dir + '/vector.json'
                    return tile, json_file, tile_name

def visualize_labels_naip(image, json_file):
    image = np.load(image)#np.array(Image.open(image))
    with open(json_file, 'r') as f:
        vector = json.load(f)
    for cat, vals in vector.items():
        instance_id = 1
        for att in vals:
            Geotype = att['Geometry']['Type'].lower()
            for geoK, geoV in att['Geometry'].items():
                if geoK.lower() == Geotype:
                    if Geotype == 'polygon':
                        for poly in geoV:
                            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            polyG = np.array(poly, np.int32).reshape((-1, 1, 2))
                            # image = cv2.fillPoly(image, [polyG], color)
                            image = cv2.polylines(image, [polyG], isClosed=True, color=color, thickness=2)
                    elif Geotype == 'polyline':
                        polyL = np.array(geoV, np.int32).reshape((-1, 1, 2))
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        image = cv2.polylines(image, [polyL], isClosed=False, color=color, thickness=2)
                    elif Geotype == 'point':
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        image = cv2.circle(image, geoV, radius=4, color=color, thickness=-1)
            instance_id += 1
    cv2.imwrite('outs/test_N_L.png', image[:,:,::-1])
    return image

def find_sentinel(xy_coord, img_data_root):
    data_folder = img_data_root + 'sentinel2/'
    for tdir in os.listdir(data_folder):
        tsub_directory = data_folder + tdir + '/tci/'
        print(f'searching in {tsub_directory}')
        for img in os.listdir(tsub_directory):
            if img.split('.')[0] == xy_coord:
                image = tsub_directory + img
                print(f'found it... => {image}')
                cv2.imwrite('outs/test_S.png', image[:,:,::-1])
                return image
    return False

img_data_root = '' #once mario downloaded sentinel images we will easily take the .npy file of sentinel2 image under `xy_ccord``
data_folder   = '/data/satlas_pretrain/naip/'
labels        = 'labels/'

tiles = [d for d in os.listdir(data_folder) if 'tci' in d]
tile = data_folder + 'naip_2019_1500_2885_tci.npy'#random.choice(tiles)
print(f'randomly selected: {tile}')
tile, json_file, xy_coord = extract_labels_naip(tile, labels)
_ = visualize_labels_naip(tile, json_file)
# if (sentinel:=find_sentinel(xy_coord, img_data_root)):
#     print(sentinel)
