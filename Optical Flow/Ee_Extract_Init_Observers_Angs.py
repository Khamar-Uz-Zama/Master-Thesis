# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:03:18 2021

@author: Khamar Uz Zama

This program extracts magnitude and angle of movement for all initial observers

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
import time

from os.path import join

videos_folder = A_Config.videos_folder
input_flow_folder = A_Config.flow_folder
initial_observer_angs_folder = A_Config.initial_observer_angs_folder

video_file = A_Config.video_file 
mask_video_file = A_Config.mask_video_file
input_flows_file = A_Config.flows_file
angs_file_initial_observers = A_Config.angs_file_initial_observers
mags_file_initial_observers = A_Config.mags_file_initial_observers
output_all_peak_results_folder = A_Config.all_peak_results_folder
this_video_results_folder = video_file[0:-4]


def get_intersection_points(full_boundary_matrix,line):
        # Finds the intersection between a line and the liver boundary 
        # and returns the  coordinates of intersection
        
        # lines = np.vstack([center[0], center[1], 700, center[1]]).T.reshape(-1, 2, 2)
        lines = np.vstack(line).T.reshape(-1, 2, 2)
        
        tmp_line = np.zeros_like(full_boundary_matrix)
        tmp_line = cv2.polylines(tmp_line, lines, 0, (255,255,255))

        zLogicalAnd = np.logical_and(tmp_line,full_boundary_matrix)
        reqd_boundary_indices = np.where(zLogicalAnd == [True])
        
         
        return reqd_boundary_indices

def initialize_plot_lines(nr_of_lines):
    # Initialize plot.
    fig_ang, ax_ang = plt.subplots()
    ax_ang.set_xlabel('Frame')
    ax_ang.set_ylabel('Angle of Movement')
    ax_ang.set_title('Angle of Movement vs Frame for '+video_file[:-4])
    ax_ang.set_xlim(0, flow_length)
    ax_ang.set_ylim(0, 400)
    # ax_ang.set_ylim(0, 10)
    
    ang_lines = [0] * nr_of_lines
    for i in range(nr_of_lines):
        ang_lines[i], = ax_ang.plot(np.arange(flow_length), np.zeros((flow_length,1)), c=patch_colors_RGB[i], lw=3, label='Obsv-'+str(i+1))
    ax_ang.legend()
    
    
    plt.ion()
    plt.show()
    
    return fig_ang, ang_lines
    

# Load Flow data
with open(join(os.getcwd(),input_flow_folder,input_flows_file), "rb") as fp:
    flows = pickle.load(fp)
    
# Calculate Boundary data
reqd_boundary_points, reqd_boundary_rects, full_boundary_points, full_boundary_matrices = C_Get_Liver_Boundary.get_liver_boundaries()
    
vertical_centers,vertical_patches = D_Create_Observers.create_grid(reqd_boundary_points,full_boundary_matrices,vertical_only=True)    

flow_length = len(flows)

    
frame_cnt = 0
vid_capture = cv2.VideoCapture(cv2.samples.findFile(join(os.getcwd(),videos_folder,video_file)))

frame_avg_patch_angs = [0.] * len(vertical_centers)
frame_avg_patch_angs = np.array(frame_avg_patch_angs)

frame_avg_patch_mags = [0.] * len(vertical_centers)
frame_avg_patch_mags = np.array(frame_avg_patch_mags)

# Add all patches data in this array
# all_frames_all_patches_angs = [0.] * flow_length
all_frames_all_patches_angs = [[0.]*len(vertical_centers)]*flow_length
all_frames_all_patches_angs = np.array(all_frames_all_patches_angs,dtype = object)

all_frames_all_patches_mags = [[0.]*len(vertical_centers)]*flow_length
all_frames_all_patches_mags = np.array(all_frames_all_patches_angs,dtype = object)

# BGR
patch_colors_BGR = [(0, 0, 255),    # Red
                   (0, 255, 0),     # Green
                   (255, 0, 0),     # Blue
                   (0, 200, 200),
                   (200, 200, 0), 
                   (200, 0, 200),
                   (50, 100, 200),
                   (100, 200, 50), 
                   (200, 50, 100),
                   (80, 50, 100)
                   ]
# Convert to RGB
patch_colors_RGB = [(color[2],color[1],color[0]) for color in patch_colors_BGR]    
patch_colors_RGB = np.array(patch_colors_RGB,float) / 255

ret, frame1 = vid_capture.read()
fig_ang, ang_lines = initialize_plot_lines(nr_of_lines=len(vertical_centers))

while(1):
    ret, frame2 = vid_capture.read()
    if not ret:
        break

    flow = flows[frame_cnt]
    this_frame_all_angs = np.empty_like(vertical_patches) 
    patch_cnt = 0
    for center,patch in zip(vertical_centers,vertical_patches):
        x = patch[0]
        y = patch[1]
        
        # angle returned is in radians... ang*180/np.pi gives angle in degree
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])            
        # Change from coordinate system to array. The indices are reversed        
        angxy = ang[y,x]
        magxy = mag[y,x]


        # average patch angle in degrees                
        angxy = (angxy*180)/np.pi
        frame_avg_patch_angs[patch_cnt] = np.average(angxy)
        # frame_avg_patch_angs[patch_cnt] = angxy
        frame_avg_patch_mags[patch_cnt] = np.average(magxy)
       
        # Below for visualization
        vis = frame1    
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
       
        # Angles are required for calculating average slope while plotting lines
        slopes = []
        for (x1, y1), (x2, y2) in lines:
            # Draw initial patch p at frame t
            cv2.circle(vis, (x1, y1), 1, patch_colors_BGR[patch_cnt], -1)
            # Draw final patch p at frame t+1
            # cv2.circle(vis, (x2, y2), 1, (255, 0, 0), -1)
            slopes.append((y2-y1)/(x2-x1))
        
       
        # To plot lines with average slope of patch
        slope = np.average(slopes)
        # x2 = 100
        # y2 = np.int32(slope *(x2-center[0])+center[1])       
        # # myPoints = np.int32([center,[x2,y2]])
        # cv2.polylines(vis,[myPoints] , 0, (0,255,0))

        # To plot line that intersects with liver - User this if you want lines on either side of the patch with the average slope of the patch
        x2 = 100
        y2 = np.int32(slope *(x2-center[0])+center[1])
        
        intersection_points = get_intersection_points(full_boundary_matrices[frame_cnt],[center[0], center[1], x2,y2])
        if(len(intersection_points[0]) > 0):
            cv2.circle(vis, (intersection_points[1][0], intersection_points[0][0]), 1, (255, 255, 255), -1)
            
            myPoints = np.int32([center,[intersection_points[1][0],intersection_points[0][0]]])
            #cv2.polylines(vis,[myPoints] , 0, patch_colors_BGR[patch_cnt])


        x2 = 800
        y2 = np.int32(slope *(x2-center[0])+center[1])
        intersection_points = get_intersection_points(full_boundary_matrices[frame_cnt],[center[0], center[1],  x2,y2])
        if(len(intersection_points[0]) > 0):
            cv2.circle(vis, (intersection_points[1][0], intersection_points[0][0]), 1, (255, 255, 255), -1)
            
            myPoints = np.int32([center,[intersection_points[1][0],intersection_points[0][0]]])
            #cv2.polylines(vis,[myPoints] , 0, patch_colors_BGR[patch_cnt])     

        #ang_lines[patch_cnt].set_ydata(all_frames_all_patches_angs[:,patch_cnt])
        #plot all magnitudes
        #fig_ang.canvas.draw()
        patch_cnt += 1

    all_frames_all_patches_angs[frame_cnt] = frame_avg_patch_angs
    all_frames_all_patches_mags[frame_cnt] = frame_avg_patch_mags
    frame_cnt += 1
    frame1 = frame2    
    for point in zip(full_boundary_points[frame_cnt][0],full_boundary_points[frame_cnt][1]):
        cv2.circle(vis,tuple(point),1,(255,255,255))        
    
    for point in zip(reqd_boundary_points[frame_cnt][0],reqd_boundary_points[frame_cnt][1]):
        cv2.circle(vis,tuple(point),1,(255,0,255))
        
    cv2.imshow('frame',vis)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if frame_cnt == 28:
        random_frame = vis
        # Save random image for reference
        if not os.path.isdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder)):
            os.mkdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder))
        plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'Init_Obsvs.png'
        cv2.imwrite(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name,),vis)        
    
    

save_title = "_AOM"
plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+save_title
plot_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)
fig_ang.savefig(plot_path)    
    
if(A_Config.flg_save_flow_data_of_initial_observers):
    with open(join(os.getcwd(),initial_observer_angs_folder,angs_file_initial_observers), "wb") as fp:
        pickle.dump(all_frames_all_patches_angs, fp)    
    print("Saved ang data of initial observers")
    with open(join(os.getcwd(),initial_observer_angs_folder,mags_file_initial_observers), "wb") as fp:
        pickle.dump(all_frames_all_patches_mags, fp)    
    print("Saved mag data of initial observers")
    
    
angles = D_Create_Observers.read_angles_of_init_observers()
for a in angles:    
    print(a)