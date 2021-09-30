#!/usr/bin/env python
# coding: utf-8

# ## Evaluate Model

# In[1]:


import sys
import os
from tqdm import tqdm
import json
import geojson, sys
from rasterio.features import shapes
import shapely.geometry
from shapely.geometry.polygon import Polygon
import fiona
import rasterio
import rasterio.mask
from math import *

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings
warnings.filterwarnings('ignore')


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('circledetectionModel.pt', map_location=device)
model.eval()
model.to(device)


# In[3]:


def get_prediction(img_path, confidence, min_confidence=0.3, step=0.05):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
        - min_confidence - if boxes aren't found confidence is reduced by step until reaches min_confidence
        - step - reduction of confidence for each try
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
          ie: eg. segment of circle is made 1 and rest of the image is made 0
    
    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_boxes = []
    
    while len(pred_boxes) == 0 and confidence >= min_confidence:
        if len([x for x in pred_score if x>confidence]) > 0:
            pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
            pred_boxes = pred_boxes[:pred_t+1]
        confidence = confidence - step
        
    if len(pred_boxes) == 0 and len(pred_score) > 0:
        pred_t = [pred_score.index(x) for x in pred_score if x>0][-1]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_boxes = pred_boxes[:pred_t+1]
    
    
    return pred_boxes


# ## Generate Mask Files

# In[10]:


TEST = True

ROOT_DATA_DIR = "../full/train/"
ROOT_WDATA_DIR = ""
if TEST:
    ROOT_DATA_DIR = "../full/test/"
    ROOT_WDATA_DIR = "./wdata/"

ROOT_DATA_DIR = sys.argv[1]
ROOT_WDATA_DIR = sys.argv[2]

print(ROOT_DATA_DIR)
print(ROOT_WDATA_DIR)

if not os.path.exists(ROOT_WDATA_DIR + "/solution/"):
    os.makedirs(ROOT_WDATA_DIR + "/solution/")
if not os.path.exists(ROOT_WDATA_DIR + "/tmp_masks/"):
    os.makedirs(ROOT_WDATA_DIR + "/tmp_masks/")
    
def get_directories(path):
    directories = []
    for x in os.walk(path):
        img_id = x[0][len(path):]
        if img_id != "":
            directories.append(img_id)
    return list(sorted(directories))

directories = get_directories(ROOT_DATA_DIR)
total_files = len(directories)

print(total_files)
print(directories[:10])


# In[11]:


for img_id in tqdm(directories):
    try:
        img_path = "".join([ROOT_DATA_DIR, img_id, "/", img_id, "_PAN.tif"])
        boxes = get_prediction(img_path, confidence=0.90, min_confidence=0.50, step=0.05)

        img = cv2.imread(img_path)
        newimg = np.zeros_like(img).astype(np.uint8)
        for i in range(len(boxes)):
            box = boxes[i]
            xmin = box[0][0]
            ymin = box[0][1]
            xmax = box[1][0]
            ymax = box[1][1]
            radius = round((np.minimum(xmax-xmin,ymax-ymin)/2) * 0.9)
            newimg = cv2.circle(newimg, (round((xmin+xmax)/2),round((ymin+ymax)/2)), radius, (255, 255, 255), -1)

        if TEST:
            cv2.imwrite("".join([ROOT_WDATA_DIR, "/tmp_masks/", img_id, "_BOX_S.tif"]), newimg)
        else:
            cv2.imwrite("".join([ROOT_DATA_DIR, img_id, "/", img_id, "_BOX_S.tif"]), newimg)
    except:
        print(img_id)
        img_path = "".join([ROOT_DATA_DIR, img_id, "/", img_id, "_PAN.tif"])
        img = cv2.imread(img_path)
        newimg = np.zeros_like(img).astype(np.uint8)
        if TEST:
            cv2.imwrite("".join([ROOT_WDATA_DIR, "/tmp_masks/", img_id, "_BOX_S.tif"]), newimg)
        else:
            cv2.imwrite("".join([ROOT_DATA_DIR, img_id, "/", img_id, "_BOX_S.tif"]), newimg)


# ## Generate GeoJson Files

# In[12]:


def compact(geometries):
    compact_geoms = []
    for p in geoms:
        compact_polygons = []
        polygons = p['geometry']['coordinates']
        for polygon in polygons:
            polygon = Polygon(polygon)
            polygon = polygon.convex_hull
            while True:
                feature_geom = Polygon(polygon)
                feature = feature_geom.area
                unit_circle = feature_geom.length ** 2 / (4 * pi)
                compactness = feature / unit_circle
                if feature_geom.is_valid and compactness >= 0.85:
                    break
                polygon = polygon.buffer(1)
            
            compact_polygons.append(shapely.geometry.mapping(polygon))
        
        for cp in compact_polygons:
            compact_geoms.append({'type': 'Feature', 'properties': {}, 'geometry': cp})
    
    return compact_geoms


# In[13]:



for img_id in tqdm(directories):
    
    img_path = "".join([ROOT_DATA_DIR, img_id, "/", img_id, "_PAN.tif"])
    mask_path = "".join([ROOT_DATA_DIR, img_id, "/", img_id, "_BOX_S.tif"])
    if TEST:
        img_path = "".join([ROOT_DATA_DIR, img_id, "/", img_id, "_PAN.tif"])
        mask_path = "".join([ROOT_WDATA_DIR, "/tmp_masks/", img_id, "_BOX_S.tif"])
    
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask = np.array(mask)[:,:,1]

    geoms = None
    geojson_file = ""
    
    try:

        with rasterio.Env():
            with rasterio.open(img_path) as original:
                with rasterio.open(mask_path) as src:
                    image = src.read(1) # first band
                    results = (
                    {'type': 'Feature', 'properties': {}, 'geometry': s}
                    for i, (s, v) 
                    in enumerate(
                        shapes(image, mask=mask, connectivity=8, transform=original.transform)))

                    geoms = list(results)
                    compact_geoms = compact(geoms)

                geojson_file = {
                "type": "FeatureCollection",
                "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:"+str(original.crs).replace(":","::")} },
                "features": compact_geoms
                }
    except:
        geojson_file = {
                "type": "FeatureCollection",
                "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:"+str(original.crs).replace(":","::")} },
                "features": []
                }
    
    if TEST:
        with open("".join([ROOT_WDATA_DIR, "/solution/", img_id, "_anno.geojson"]), 'w') as outfile:
            json.dump(geojson_file, outfile)
    else:
        with open("".join([ROOT_DATA_DIR, img_id, "/", img_id, "_anno2.geojson"]), 'w') as outfile:
            json.dump(geojson_file, outfile)
            


# ## Test Locally

# In[14]:


CIRCULAR_THRES = 0.85
IOU_THRES = 0.5

def load_polygons(geo_json_file, debug=False):
    ret = []
    try:
        with open(geo_json_file) as f:
            data = geojson.load(f)['features']
            for i in range(len(data)):
                polygons = data[i]['geometry']['coordinates']
                for polygon in polygons:
                    feature_geom = Polygon(polygon)
                    feature = feature_geom.area
                    unit_circle = feature_geom.length ** 2 / (4 * pi)
                    compactness = feature / unit_circle

                    if feature_geom.is_valid and compactness >= CIRCULAR_THRES:
                        ret.append(feature_geom)
                    else:
                        if debug:
                            print("No compactness")
    except:
        pass

    return ret

def get_IoU(poly1, poly2):
    i = poly1.intersection(poly2).area
    u = poly1.area + poly2.area - i
    return i / u


# In[15]:


target = directories
score = 0

count_zeros = 0
predicted_polygons = []
real_polygons = []

for filename in target:
    if TEST:
        pred = load_polygons("".join([ROOT_WDATA_DIR, "/solution/", filename, "_anno.geojson"]), debug=True)
        #print("Predicted polygons: " + str(len(pred)))
    else:
        pred = load_polygons("".join([ROOT_DATA_DIR, filename, "/", filename, "_anno2.geojson"]), debug=True)
        
        truth = load_polygons("".join([ROOT_DATA_DIR, filename, "/", filename, "_anno.geojson"]))
        
        if len(truth) == 0:
            f1 = float(len(pred) == 0)
            print("No real polygons")
        elif len(pred) == 0:
            f1 = 0
            print("No predicted polygons" + " TEST CASE: " + filename )
        elif len(pred) > 2000:
            f1 = 0
            print("More than 2000 polygons")
        else:
            truth_area = [x.area for x in truth]
            matched = [False for i in range(len(truth))]
            overlap = 0
            for pred_poly in pred:
                best_IoU, best_i = IOU_THRES, -1
                pred_poly_area = pred_poly.area
                for i in range(len(truth)):
                    if not matched[i]:
                        max_IoU = min(pred_poly_area, truth_area[i]) / max(pred_poly_area, truth_area[i])
                        if max_IoU > best_IoU:
                            cur_IoU = get_IoU(pred_poly, truth[i])
                            if cur_IoU > best_IoU:
                                best_IoU, best_i = cur_IoU, i
                if best_IoU > IOU_THRES:
                    matched[best_i] = True
                    overlap += 1
                    if overlap == len(truth):
                        break

            precision = overlap / len(pred)
            recall = overlap / len(truth)
            if overlap == 0:
                f1 = 0
            else:
                f1 = precision * recall * 2 / (precision + recall)
            if f1 < 1:
                print("TEST CASE: " + filename + " =  " + str(f1))
                predicted_polygons.append(len(pred))
                real_polygons.append(len(truth))
                if f1 == 0:
                    count_zeros += 1
        score += f1
score /= len(target)

more_predicted_polygons = 0
fewer_predicted_polygons = 0
equal_predicted_polygons = 0
for i in range(len(predicted_polygons)):
    if predicted_polygons[i] > real_polygons[i]:
        more_predicted_polygons += 1
    elif predicted_polygons[i] < real_polygons[i]:
        fewer_predicted_polygons += 1
    else:
        equal_predicted_polygons += 1

if not TEST: 
    print("More predicted polygons: " + str(more_predicted_polygons))
    print("Fewer predicted polygons: " + str(fewer_predicted_polygons))
    print("Equal predicted polygons: " + str(equal_predicted_polygons))
    print("Zeros: " + str(count_zeros)) 
    print('Final Score =', score)
    


# train Final Score = 0.9233485658262668

# In[ ]:




