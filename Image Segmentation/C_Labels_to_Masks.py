# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 19:05:03 2020

@author: Khamar Uz Zama 
https://github.com/pateldigant/SemanticAnnotator/blob/master/SementicAnnotator2.py
Annotations are extracted using VASA annotation tool

1. Reads the Annotations file
2. Creates masks with a color for each label. In this case it's 1
3. Saves them in folder

Input: 1. Annotations file
       
Output: 1. Masks are saved in 0_Data_Masks folder

"""

import pandas as pd
import cv2
import numpy as np
import argparse
import os
import time
import json

import A_Config

start_time = time.time()

input_annotations_file = A_Config.annotations_file
output_path = A_Config.masks_folder
#output_mask_shape = A_Config.imshape # change this later as mask shape should be same as image

if not os.path.exists(output_path):
    os.makedirs(output_path)

df = pd.read_csv(input_annotations_file)
filenames = df.filename.unique()

print("Creating Masks")
print("Mask Shape = ",A_Config.orig_masks_shape[0], A_Config.orig_masks_shape[1],A_Config.orig_masks_shape[2])

for filename in filenames:
    
    temp_df = df.loc[df['filename'] == filename]
    
    data = {}
    #the colors are in bgr format .... key = class eg 1 for grass , 2 for mud ,etc.
    colors = {"liver": (255,255,255)}
    
    for index, row in temp_df.iterrows():
        polygon_string = row["region_shape_attributes"]
        label_string = row["region_attributes"]
        xy_points_string = polygon_string[polygon_string.find('"all_points_x"'):polygon_string.find('}')]
        xy_points_string = xy_points_string.split('],')
        x_points = xy_points_string[0]
        ##print(x_points)
        x_points = x_points[x_points.find('[')+1:]
        y_points = xy_points_string[1]
        y_points = y_points[y_points.find('[')+1:y_points.find(']')]
        ##print(x_points)
        x_points_int = [int(x) for x in x_points.split(',')]
        y_points_int = [int(x) for x in y_points.split(',')]
        
        json_label_string = res = json.loads(label_string) 
        class_label = json_label_string["name"]
                
        #print(x_points_int,y_points_int,class_label)
        xy_pairs = []
        for xy in zip(x_points_int,y_points_int):
            xy_pairs.append(xy)
        #print(xy_pairs)
        if class_label not in data:
            data[class_label] = []
        data[class_label].append(xy_pairs)
        
    
    mask = np.zeros((A_Config.orig_masks_shape[0], A_Config.orig_masks_shape[1],A_Config.orig_masks_shape[2]))
                    
    index = -1
    for key in data.keys():
        index += 1
        list_of_points = data[key]
        for points in list_of_points:
            pts = np.array(points)
            mask = cv2.fillPoly(mask, [pts], color=colors[key])
            
    cv2.imwrite(""+ output_path + '/' + filename.split('.')[0] +'_mask'+'.jpg', mask)
    df = df[df['filename']!=filename]
    
    
end_time = time.time()
print('Processed {} files in {:.3f} seconds'.format(len(filenames) , end_time-start_time))



