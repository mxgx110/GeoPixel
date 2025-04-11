import numpy as np   
import cv2
import json
import sys
import os
import random
import subprocess
from PIL import Image
from utils import extract_labels_sen2, visualize_labels_sen2
from utils import extract_labels_naip, visualize_labels_naip

def print_stats(dir, img_data_root):
    print(dir.replace(img_data_root, ''), '\n')
    sub_dirs = [d for d in os.listdir(dir)]
    for sd in sub_dirs: # ||ir and tci for {naip}|| and ||b05,...,tci for {sentinel2}||
        num_files = len([f for f in os.listdir(dir + sd)])
        print(f'{sd + "/"} contains {num_files} number of images inside')

def image_selector(img_data_root, src):
    data_folder = img_data_root + src 

    print('Choosing a random directory: ', end='\t=>\t')

    tdirectories = [d for d in os.listdir(data_folder)]
    random_dir = random.choice(tdirectories)
    tsub_directory = data_folder + random_dir + '/'
    
    print_stats(tsub_directory, img_data_root)
    tsub_directory += 'tci/' # we check rgb images for now

    images = [d for d in os.listdir(tsub_directory)]
    image = random.choice(images)
    image = tsub_directory + image
    print(f"\nThe randomly-selected image: {image}")

    command = ["catimg", "-w", "64", image]
    subprocess.run(command)
    user_answ = input('Do you want to proceed with this image? [visual check] ')
    if user_answ.lower() == 'no':
        return False
    else:
        return image

def extract_labels(image, labels, src):
    if 'naip' in src:
        return extract_labels_naip(image, labels)
    else:
        return extract_labels_sen2(image, labels)

def visualize_labels(image, json_file, src, tlbr):
    if 'naip' in src:
        return visualize_labels_naip(image, json_file, tlbr)
    else:
        return visualize_labels_sen2(image, json_file)


if __name__ == "__main__":
    print('\t\t<<<This code takes a random naip/sent2 image and save it with its labels plotted as `test.png` in the same directory>>>\n')
    # try:
    SOURCE = 'naip/' if sys.argv[1].lower() == 'n' else 'sentinel2/' #'n' for naip and otherwise for sentinel2
    REVIEWED_IMAGES = []
    SELECTED_IMAGES = []

    #----------------------------------------------change these two lines----------------------------------------------#
    IMG_DATA_ROOT   = ''        # where naip/ and sentinel2/ are located => (for me it is same as where this .py file is located)
    LABEL_DATA_ROOT = 'labels/' # where static(s)/ are located => I assume they are named static0/ , static1/, static2/
    #----------------------------------------------change these two lines----------------------------------------------#

    while True:
        out = image_selector(img_data_root=IMG_DATA_ROOT, src=SOURCE)
        if out:
            if out in REVIEWED_IMAGES:
                continue
            image, json_file, tlbr = extract_labels(image=out, labels=LABEL_DATA_ROOT, src=SOURCE)
            REVIEWED_IMAGES.append(image)
            _ = visualize_labels(image=image, json_file=json_file, src=SOURCE, tlbr=tlbr)
            user_answ = input('Do you take this image and its labels? [label check] ')
            if user_answ == 'yes':
                SELECTED_IMAGES.append((image, json_file))
                print(f'\n\t\tyou took ##{image}## and its labels ##{json_file}##') 
                print('-'*100)
            else:
                print('-'*100)
                continue
        else:
            print('-'*100)
            continue
        print(f'Length of the selected data up to now: {len(SELECTED_IMAGES)}')
        
    # except:
    #     print('ERROR! Hahah...')
    #     print('1) Did you modify `main.py: LINE64 and LINE65` according to their comments?')
    #     print('2) Did you run the command as: `python main.py <src>` where <src> is either `n` for NAIP or `s` for SENTINEL-2 ?')
    #     print('3) Did you install `catimg`? {sudo apt install catimg}')
    #     print('* If you cant install `catimg`, comment `main.py: LINE36/37` and always press ENTER when prompted by python input()')
