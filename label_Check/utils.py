import numpy as np   
import cv2
import json
import os
import random
import subprocess
from PIL import Image 
#---------------------utils for sentinel-2 images
def extract_labels_sen2(image, labels):
    image_name = image.split('/')[-1].split('.')[0]
    for lb_dir in os.listdir(labels):
        if lb_dir.startswith('static'):
            st_folder = lb_dir + '/'
            print(f'searching in {st_folder}...')
            for dir in os.listdir(labels + st_folder):
                if dir == image_name:
                    json_file = labels + st_folder + dir + '/vector.json'
                    print(f'found it! analyzing {json_file}\n{"#"*75}')
                    return image, json_file, None

def visualize_labels_sen2(image, json_file):
    image = np.array(Image.open(image)) # 512x512
    with open(json_file, 'r') as f:
        vector = json.load(f)
    #--------------------------Static
    for cat, vals in vector.items():
        if len(vals) > 0:
            print(f'{"*"*75}\nCategory: {cat}')
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        instance_id = 1
        for att in vals: # many [Geometry, Properties] iside list vals: different instances of the same category
            Geotype  = att['Geometry']['Type'].lower() #point/polyline/polygon
            # Property = att['Properties'] #maybe later
            for geoK, geoV in att['Geometry'].items():
                if geoK.lower() == Geotype:
                    #scale of coordinates is 8kx8k => same as NAIP => makes sense
                    if Geotype in ['polygon', 'polyline']:
                        print(cat, instance_id, Geotype)
                        geoV = geoV[0] if Geotype=='polygon' else geoV # [0] is removing the extra [] around the polygone
                        polyX = list(map(lambda x: [x[0]//16, x[1]//16], geoV)) # make 8kx8k, 512x512 only for sentinel-2 images
                        polyX = np.array(polyX, np.int32).reshape((-1, 1, 2))

                        isClosed = True if Geotype=='polygon' else False
                        image = cv2.polylines(image, [polyX], isClosed=isClosed, color=color, thickness=2)
                    elif Geotype == 'point':
                        print(cat, instance_id, Geotype)
                        point = (geoV[0]//16, geoV[1]//16) # make 8kx8k, 512x512 only for sentinel-2 images
                        image = cv2.circle(image, point, radius=4, color=color, thickness=-1)

            instance_id += 1
    #-------------------------Dynamic
    cv2.imwrite('test.png', image[:,:,::-1])
    return image
    # try:
    #     subprocess.run(["mv", json_file.replace('/vector.json', '/raster0.png'), json_file.replace('/vector.json', '/raster1.png'), '/home/ghahramani/GeoPixel/label_Check/'], check=True) #move raster0/1 here
    # except:
    #     print('No raster.png file')


#---------------------utils for NAIP images
def extract_labels_naip(image, labels):
    #image: 12345_67890.png => 12345//16=771, 67890//16=4243 => 771_4243 {:=image_name}
    first, second = list(map(lambda x: str(int(x)//16), image.split('/')[-1].split('.')[0].split('_')))
    tl = list(map(lambda x: 512*(int(x)%16), image.split('/')[-1].split('.')[0].split('_'))) #512*16=8192
    br = [tl[0]+512, tl[1]+512]
    image_name = first + '_' + second
    for lb_dir in os.listdir(labels):
        if lb_dir.startswith('static'):
            st_folder = lb_dir + '/'
            print(f'searching in {st_folder}...')
            for dir in os.listdir(labels + st_folder):
                if dir == image_name:
                    json_file = labels + st_folder + dir + '/vector.json' #771_4243/vector.json
                    print(f'found it! analyzing {json_file}\n{"#"*75}')
                    return image, json_file, (tl, br)

def offset_modifier(inp, tlbr):
    tl, br = tlbr
    inp = list(filter(lambda x: (tl[0] <= x[0] <= br[0]) and (tl[1] <= x[1] <= br[1]), inp)) # take the crop
    inp = list(map(lambda x: [x[0]-tl[0], x[1]-tl[1]], inp)) # offset : we need the coordinates in the 512*512 scale and not 8kx8k
    return inp

def visualize_labels_naip(image, json_file, tlbr):
    image = np.array(Image.open(image)) # 512x512
    with open(json_file, 'r') as f:
        vector = json.load(f)
    #--------------------------static
    for cat, vals in vector.items():
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        instance_id = 1
        for att in vals: # many [Geometry, Properties] iside list vals: different instances of the same category
            Geotype  = att['Geometry']['Type'].lower() #point/polyline/polygon
            # Property = att['Properties'] #maybe later
            for geoK, geoV in att['Geometry'].items():
                if geoK.lower() == Geotype:
                    #scale of coordinates is 8kx8k => same as NAIP => makes sense
                    if Geotype in ['polygon', 'polyline']:
                        geoV = geoV[0] if Geotype=='polygon' else geoV # [0] is removing the extra [] around the polygone

                        polyX = offset_modifier(geoV, tlbr)
                        if len(polyX) > 0:
                            print(f'{"*"*75}\nCategory: {cat}')
                            print(cat, instance_id, Geotype)
                            polyX = np.array(polyX, np.int32).reshape((-1, 1, 2))
                            isClosed = True if Geotype=='polygon' else False
                            image = cv2.polylines(image, [polyX], isClosed=isClosed, color=color, thickness=2)

                    elif Geotype == 'point':
                        point = offset_modifier([geoV], tlbr) # make 8kx8k, 512x512 only for sentinel-2 images
                        if len(point) > 0:
                            print(f'{"*"*75}\nCategory: {cat}')
                            print(cat, instance_id, Geotype)
                            point = point[0]
                            image = cv2.circle(image, point, radius=4, color=color, thickness=-1)

            instance_id += 1
    #----------------------Dynamic
    from dynamic_naip_T import extract_labels_naip_D, visualize_labels_naip_D
    labels = json_file.split('/')[0]+'/'
    _, json_dir, tile_name = extract_labels_naip_D(tile=json_file, labels=labels, main=True)
    print(json_dir)
    for json_sdir in os.listdir(json_dir):
        print(json_sdir)
        if '.json' in json_sdir:
            image = visualize_labels_naip_D(image, json_dir + '/vector.json', main=True, tile_name=tile_name)
            break
        image = visualize_labels_naip_D(image, json_dir + '/' + json_sdir + '/vector.json', main=True, tile_name=tile_name)

    cv2.imwrite('test.png', image[:,:,::-1])
    return image
    # try:
    #     subprocess.run(["mv", json_file.replace('/vector.json', '/raster0.png'), json_file.replace('/vector.json', '/raster1.png'), '/home/ghahramani/GeoPixel/label_Check/'], check=True) #move raster0/1 here
    # except:
    #     print('No raster.png file')