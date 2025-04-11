import numpy as np   
import cv2
import json
import os
import random
import subprocess
from PIL import Image 
import sys

def offset_modifier(inp, tlbr):
    tl, br = tlbr
    inp = list(filter(lambda x: (tl[0] <= x[0] <= br[0]) and (tl[1] <= x[1] <= br[1]), inp)) # take the crop
    inp = list(map(lambda x: [x[0]-tl[0], x[1]-tl[1]], inp)) # offset : we need the coordinates in the 512*512 scale and not 8kx8k
    return inp

def extract_labels_naip_D(tile, labels, main=False):    
    tile_name = (tile.split('/')[-1].split('.')[0])[10:19] if not main else tile.split('/')[-2]
    for lb_dir in os.listdir(labels):
        if lb_dir.startswith('dynamic'):
            st_folder = lb_dir + '/'
            for dir in os.listdir(labels + st_folder):
                # print(dir, "===", tile_name)
                if dir == tile_name:
                    json_dir = labels + st_folder + dir
                    print(f'found the directory... {json_dir}')
                    return tile, json_dir, tile_name
    print('COULD NOT FIND IT.....................................................')

def visualize_labels_naip_D(image, json_file, main=False, tile_name=None):
    # image = np.load(image)#np.array(Image.open(image))
    if main:
        first, second = tile_name.split('_')
        tl = 512*int(first), 512*int(second)
        br = [tl[0]+512, tl[1]+512]
        tlbr = (tl, br)

    with open(json_file, 'r') as f:
        vector = json.load(f)
    print(vector)
    for cat, vals in vector.items():
        if cat == 'metadata':
            continue
        instance_id = 1
        for att in vals:
            Geotype = att['Geometry']['Type'].lower()
            for geoK, geoV in att['Geometry'].items():
                if geoK.lower() == Geotype:
                    print(cat, instance_id, Geotype)
                    print(geoV)
                    print('-'*75)
                    if Geotype == 'polygon':
                        if main:
                            geoV = offset_modifier(geoV, tlbr)
                        for poly in geoV:
                            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            polyG = np.array(poly, np.int32).reshape((-1, 1, 2))
                            # image = cv2.fillPoly(image, [polyG], color)
                            image = cv2.polylines(image, [polyG], isClosed=True, color=color, thickness=2)
                    elif Geotype == 'polyline':
                        if main:
                            geoV = offset_modifier(geoV, tlbr)
                        polyL = np.array(geoV, np.int32).reshape((-1, 1, 2))
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        image = cv2.polylines(image, [polyL], isClosed=False, color=color, thickness=2)
                    elif Geotype == 'point':
                        if main:
                            geoV = offset_modifier([geoV], tlbr)
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        image = cv2.circle(image, geoV, radius=4, color=color, thickness=-1)
            instance_id += 1
    
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


if __name__ == "__main__":
    # img_data_root = '' #once mario downloaded sentinel images we will easily take the .npy file of sentinel2 image under `xy_ccord``
    data_folder   = '/data/satlas_pretrain/naip/'
    labels        = 'labels/'

    tiles = [d for d in os.listdir(data_folder) if 'tci' in d]
    tile = data_folder + 'naip_2016_1000_2185_tci.npy'#random.choice(tiles)#'naip_2016_2199_3033_tci.npy'
    print(f'randomly selected: {tile}')
    tile, json_dir, xy_coord = extract_labels_naip_D(tile, labels)

    tile_image = np.load(tile)

    for json_sdir in os.listdir(json_dir):
        if '.json' in json_sdir:
            tile_image = visualize_labels_naip_D(tile_image, json_dir + '/vector.json')
            break
        tile_image = visualize_labels_naip_D(tile_image, json_dir + '/' + json_sdir + '/vector.json')

    cv2.imwrite('outs/test_N_L.png', tile_image[:,:,::-1])

    # if (sentinel:=find_sentinel(xy_coord, img_data_root)):
    #     print(sentinel)
