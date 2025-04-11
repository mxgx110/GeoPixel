import json
import sys
import os
import numpy as np   
import cv2
from PIL import Image
import random

def take_crop(i, j, naip_res):
    tl = [i*naip_res, j*naip_res]
    br = [(i+1)*naip_res, (j+1)*naip_res]
    return (tl, br)

def offset_modifier(inp, crop, map=True):
    tl, br = crop
    filtered = list(filter(lambda x: (tl[0] <= x[0] <= br[0]) and (tl[1] <= x[1] <= br[1]), inp)) # take the crop
    if map:
        filtered = list(map(lambda x: [x[0]-tl[0], x[1]-tl[1]], filtered)) # offset : we need the coordinates in the 512*512 scale and not 8kx8k
    return np.array(filtered, np.int32)

def vector_corrector(vector, res):
    def subdivide(line, res):
        #line: Nx2
        out_line = []
        for i in range(len(line)-1):
            firstP  = line[i] #2D point
            secondP = line[i+1] #2D point
            out_line.append(firstP) # first appending

            # compute the line equation
            inf_slope = False
            if secondP[0] == firstP[0]:
                inf_slope = True
            else:
                m = (secondP[1]-firstP[1]) / (secondP[0]-firstP[0]) #slope
                h = firstP[1] - m*firstP[0] #y-intercept

            kx1, kx2 = firstP[0]//res, secondP[0]//res
            ky1, ky2 = firstP[1]//res, secondP[1]//res
            Mx, My = abs(kx1 - kx2), abs(ky1 - ky2)

            for sx in range(Mx):
                boundX = res * (min(kx1, kx2) + sx + 1)
                boundY = m * boundX + h
                out_line.append([boundX, boundY])
            for sy in range(My):
                boundY = res * (min(ky1, ky2) + sy + 1)
                if inf_slope: # only in My inf_slope might happen
                    boundX = firstP[0]
                else:
                    boundX = (boundY - h) / m
                out_line.append([boundX, boundY])

        out_line.append(line[-1]) # also include the last point
        return np.array(out_line, np.int32)

    for cat, vals in vector.items():
        for idx, att in enumerate(vals):
            Geotype  = att['Geometry']['Type'].lower() #point/polyline/polygon
            for geoK, geoV in att['Geometry'].items():
                if geoK.lower() == Geotype:
                    if Geotype == 'polyline' or Geotype == 'polygon': #geoV: Nx2
                        if Geotype == 'polyline': print('Yes')
                        # else: print('no')
                        vector[cat][idx]['Geometry'][geoK] = subdivide(geoV, res)
    return vector

def label_filter(json_file, debug=False):
    tile_res = 8192
    naip_res = 512
    size = tile_res // naip_res #16

    with open(json_file, 'r') as f:
        vector = json.load(f)
        vector = vector_corrector(vector=vector, res=naip_res) #handles Problem 1)

    counts = {}
    init = {k:0 for (k, _) in vector.items()}
    for i in range(size):
        for j in range(size):
            counts[(i, j)] = init.copy()

    size = 5
    for i in range(size): #col
        for j in range(size): #row
            crop = take_crop(i, j, naip_res)
            for cat, vals in vector.items(): 
                for att in vals:
                    Geotype  = att['Geometry']['Type'].lower() #point/polyline/polygon
                    for geoK, geoV in att['Geometry'].items():
                        if geoK.lower() == Geotype:
                            if Geotype == 'polygon':
                                #for polygon: geoV: KxNx2 : N points for each polygon. K polygons that might be nested.
                                for polyG in geoV:
                                    filtered = offset_modifier(polyG, crop) # Nx2
                                    if len(filtered) > 0:
                                        counts[(i, j)][cat] += 1  
                                        ##
                                        if (i == col) and (j == row) and type(debug) == np.ndarray:
                                            debug = cv2.fillPoly(debug, [filtered], (0,0,0))
                                            cv2.putText(debug, str(counts[(i, j)][cat]), tuple(filtered[0]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1)
                                            cv2.putText(debug, cat, tuple(filtered[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1)
                                        ##                                 
                            elif Geotype == 'polyline':
                                #for polyline: geoV: Nx2 : N points for each polygon. There is no nesting.
                                filtered = offset_modifier(geoV, crop) # Nx2
                                if len(filtered) > 0:
                                    counts[(i, j)][cat] += 1
                                    ##
                                    if (i == col) and (j == row) and type(debug) == np.ndarray:
                                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                                        debug = cv2.polylines(debug, [filtered], isClosed=False, color=color, thickness=3)
                                        cv2.putText(debug, str(counts[(i, j)][cat]), filtered[len(filtered)//2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                    ##
                            elif Geotype == 'point':
                                #for point: geoV: (2, ) : One single point. That is why we call offset_modifier(.) over [geoV] (shape with [] :(1x2))
                                filtered = offset_modifier([geoV], crop)
                                if len(filtered) > 0:
                                    counts[(i, j)][cat] += 1 
                                    ##
                                    if (i == col) and (j == row) and type(debug) == np.ndarray:
                                        #filtered: 1x2
                                        debug = cv2.circle(debug, filtered[0], radius=4, color=(255,255,255), thickness=4)
                                        cv2.putText(debug, str(counts[(i, j)][cat]), tuple(filtered[0]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    ##
    return counts, debug

def counts2matrix(counts, naive=True):
    # naive=False must be implemented
    # handles Problem 2)
    size = int(np.sqrt(len(counts))) # 16 in our case
    matrix = np.zeros((size, size))
    for (col, row), c_dict in counts.items():
        nums = 0
        for cat in c_dict:
            nums += c_dict[cat]
        matrix[col][row] = nums
    return matrix

if __name__ == "__main__":

    #----if you plan to debug
    IMAGE_NAME = f'/home/ghahramani/GeoPixel/label_Check/outs/test_N_L.png' #1500_2885
    col, row = random.randint(0, 15), random.randint(0, 15)
    print(f'col: {col} --- row: {row}')
    cell_image = np.array(Image.open(IMAGE_NAME))[512*row:512*(row+1),512*col:512*(col+1),::-1]
    cv2.imwrite('cell_image.png', cell_image)


    LABEL_DIR = '/data/satlas_pretrain/labels_static/static/' 
    XXXXYYYY  = sys.argv[1]
    json_file = LABEL_DIR + XXXXYYYY + '/vector.json'
    counts, debug = label_filter(json_file=json_file, debug=cell_image.copy())
    cv2.imwrite('debug.png', debug)
    #-----test: only for 1500-2885
    # print('-'*75)
    # c_matrix = counts2matrix(counts, naive=True)
    # print(c_matrix[col, row])
    # for cat, num in counts[(col, row)].items():
    #     if num != 0:
    #         print(cat, ' : ', num)
















# Probelm 1): Since roads are represented as polylines, the next point in the line might fall outside the image boundaries. 
            # As a result, the segment between the last visible point and the image edge gets omitted. 
            # Solution: we have to modify vector.json files from source. We cannot do it after taking a tile image.
# Problem 2): Some categories corresponds to the same object and when counting, we should not count them twice. Like {building | ms_building}.
# Problem 3): How is it possible that ms_building are sometimes more than buildings? (col:13 --- row:14)
# Problem 4): Sometimes a road appears to connect to itself, even though it is not actually connected within the tile. (col:6 --- row:9)
            # Solution: This happens exactly due to Problem 1. When intermediate road points fall outside the image, the next available point 
            # connects to the last visible one. Solving Problem 1 will automatically fix this as well.