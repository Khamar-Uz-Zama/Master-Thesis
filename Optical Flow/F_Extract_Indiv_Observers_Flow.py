# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 14:53:17 2021

@author: Khamar Uz Zama

Description: This program:
                1. takes output calculated from optical flow
                2. Gets required boundary of liver from C_Get_Liver_Boundary
                3. Loads the video file for visualization
                4. Gets center and patches from D_Create_Patches using the Angle from Movement determined from Step 2
                5. Extracts the flow data of each patch
                6. Saves them in A_Config.output_individual_patch_flows_folder
                7. The flow outputs are then used to extract peak points and plot box plots

Check:
    1. Check if the file name is correct
    2. Check if flow input is calculated properly
    3. Check the size and location of each patch

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import A_Config
import C_Get_Liver_Boundary
import D_Create_Observers
import os
import datetime

from os.path import join

#### Constants ######

videos_folder = A_Config.videos_folder
input_flow_folder = A_Config.flow_folder
output_individual_patch_flows_folder = A_Config.individual_patch_flows_folder

video_file = A_Config.video_file 
mask_video_file = A_Config.mask_video_file
input_flows_file = A_Config.flows_file
output_individual_patch_flow = A_Config.patch_flows_file
output_all_peak_results_folder = A_Config.all_peak_results_folder
this_video_results_folder = video_file[0:-4]

# Load Flow data
with open(join(os.getcwd(),input_flow_folder,input_flows_file), "rb") as fp:
    flows = pickle.load(fp)
    
# Calculate Boundary data
reqd_boundary_points, reqd_boundary_rects, full_boundary_points, full_boundary_matrices = C_Get_Liver_Boundary.get_liver_boundaries()
    
grid_centers,grid_patches = D_Create_Observers.create_grid(reqd_boundary_points,full_boundary_matrices)    

flow_length = len(flows)

    
frame_cnt = 0
vid_capture = cv2.VideoCapture(cv2.samples.findFile(join(os.getcwd(),videos_folder,video_file)))


avg_patch_mag = [[0.] * len(grid_centers)] * flow_length
avg_patch_mag = np.array(avg_patch_mag)

# Add all patches data in this array
all_patches_mag = [] * flow_length
#all_patches_mag = np.array(all_patches_mag)

# BGR
patch_colors = [(0, 0, 255), # Red
                (0, 255, 0), # Green
                (255, 0, 0),  # Blue
                (0, 200, 200),
                (200, 200, 0), 
                (200, 0, 200),
                (50, 100, 200),
                (100, 200, 50), 
                (200, 50, 100),
                (80, 50, 100)
                ]
ret, frame1 = vid_capture.read()

while(1):
    ret, frame2 = vid_capture.read()
    if not ret:
        break 

    flow = flows[frame_cnt]
    this_frame_all_mags = np.empty_like(grid_centers) 
    for row_id in range(len(this_frame_all_mags)):
        this_row_centers = grid_centers[row_id]
        this_row_patches = grid_patches[row_id]
        
        this_row_centers = [x for x in this_row_centers if not isinstance(x, int)]
        this_row_patches = [x for x in this_row_patches if not isinstance(x, int)]        
        
        col_id = 0
        for center,patch in zip(this_row_centers,this_row_patches):
            if(center == 0):
                this_frame_all_mags[row_id][col_id] = 0
                col_id += 1
                continue
            x = patch[0]
            y = patch[1]
            
            # angle returned is in radians... ang*180/np.pi gives angle in degree
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])            
            # Change from coordinate system to array. The indices are reversed        
            magxy = mag[y,x]
            this_frame_all_mags[row_id][col_id] = magxy

            # Below for visualization
            vis = frame1    
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines)
            
            # Angles are required for calculating average slope while plotting lines
            if(col_id >= -1):
                for (x1, y1), (x2, y2) in lines:
                    # Draw patch p at frame t
                    cv2.circle(vis, (x1, y1), 1, patch_colors[row_id], -1)
                
            # Plot line connecting first and last patch col wise
            x1,y1 = this_row_centers[0][0], this_row_centers[0][1]
            x2,y2 = this_row_centers[-1][0], this_row_centers[-1][1]
            myPoints = np.int32([[x1,y1],[x2,y2]])
            cv2.polylines(vis,[myPoints] , 0, patch_colors[row_id])
            # patch_cnt += 1

            col_id += 1
    for point in zip(full_boundary_points[frame_cnt][0],full_boundary_points[frame_cnt][1]):
        cv2.circle(vis,tuple(point),1,(255,255,255))        
    
    for point in zip(reqd_boundary_points[frame_cnt][0],reqd_boundary_points[frame_cnt][1]):
        cv2.circle(vis,tuple(point),1,(255,0,255))

    reqd_boundary_points
    all_patches_mag.append(this_frame_all_mags)
    cv2.imshow('frame',vis)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if frame_cnt == 28:
        # Save random image for reference
        if not os.path.isdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder)):
            os.mkdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder))
        plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'All_Obsvs.png'
        cv2.imwrite(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name,),vis)        
        
    frame_cnt += 1
    frame1 = frame2

    
    
# if(A_Config.flg_save_flow_data_of_patches):

#     with open(join(os.getcwd(),output_individual_patch_flows_folder,output_individual_patch_flow), "wb") as fp:
#         pickle.dump(all_patches_mag, fp)    
#     print("Saved flow data of individual observers")
