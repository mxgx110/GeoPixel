import numpy as np   
import cv2
import json
import os
import random
import subprocess
from PIL import Image 
import sys

def extract_labels_naip(image, labels):
    image_name = image.split('.')[0].split('/')[-1]
    for lb_dir in os.listdir(labels):
        if lb_dir.startswith('static'):
            st_folder = lb_dir + '/'
            for dir in os.listdir(labels + st_folder):
                if dir == image_name:
                    json_file = labels + st_folder + dir + '/vector.json'
                    return image, json_file

def visualize_labels_naip(image, json_file):
    image = np.array(Image.open(image))
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
                            polyX = np.array(poly, np.int32).reshape((-1, 1, 2))
                            image = cv2.polylines(image, [polyX], isClosed=True, color=color, thickness=2)
                    elif Geotype == 'polyline':
                        polyX = np.array(geoV, np.int32).reshape((-1, 1, 2))
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        image = cv2.polylines(image, [polyX], isClosed=False, color=color, thickness=2)
                    elif Geotype == 'point':
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        image = cv2.circle(image, geoV, radius=4, color=color, thickness=-1)
            instance_id += 1
    cv2.imwrite('composite/test_N_L.png', image[:,:,::-1])
    return image

def find_sentinel(xy_coord, img_data_root):
    data_folder = img_data_root + 'sentinel2/'
    for tdir in os.listdir(data_folder):
        tsub_directory = data_folder + tdir + '/tci/'
        print(f'searching in {tsub_directory}')
        for img in os.listdir(tsub_directory):
            # print(img.split('.')[0], '===', xy_coord)
            if img.split('.')[0] == xy_coord:
                image = tsub_directory + img
                print(f'found it... => {image}')
                cv2.imwrite('composite/test_S.png', image[:,:,::-1])
                return image
    return False

xy_coord = sys.argv[1]
image  = '/home/ghahramani/GeoPixel/label_Check/composite/' + xy_coord + '.png'
img_data_root = ''
labels = 'labels/'
image, json_file = extract_labels_naip(image, labels)
_ = visualize_labels_naip(image, json_file)
_ = find_sentinel(xy_coord, img_data_root)
