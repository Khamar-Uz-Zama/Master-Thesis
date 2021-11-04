# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 23:49:19 2021

@author: Khamar Uz Zama
This program takes a video as input and identifies the boundaries of the liver region
"""
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.feature import canny
import A_Config
import os
from os.path import join

#### Constants ######

videos_folder = A_Config.videos_folder

flow_video_file = A_Config.video_file 
mask_video_file = A_Config.mask_video_file

vid_capture = cv2.VideoCapture(cv2.samples.findFile(join(os.getcwd(),videos_folder,flow_video_file)))
msk_capture = cv2.VideoCapture(cv2.samples.findFile(join(os.getcwd(),videos_folder,mask_video_file)))

flg_vis_liv_boundary = A_Config.flg_vis_liv_boundary 
buffer_to_left,buffer_to_right = A_Config.buffer_to_left,A_Config.buffer_to_right
buffer_to_up, buffer_to_down = A_Config.buffer_to_up, A_Config.buffer_to_down

reqd_boundary_points = []
full_boundary_points = []
reqd_boundary_rects = []
full_boundary_matrices = []

# Visualization
if(flg_vis_liv_boundary):
    
    (grabbed, orig_frame) = vid_capture.read()
    (grabbed, orig_mask) = msk_capture.read()
    
    frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    
    col_length = frame.shape[1]
    row_length = frame.shape[0]
    
    # Initialize plot.
    fig, ax = plt.subplots()
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Intensity')
    
    col_line, = ax.plot(np.arange(col_length), np.zeros((col_length,1)), c='k', lw=3, label='col-wise intensity')
    col_peakPoints, = ax.plot(np.arange(col_length), np.zeros((col_length,1)), 'x', c='y', label='peaks')
    ax.set_title('Column wise intensity of segmented mask')
    ax.set_xlim(0, col_length)
    
    row_line, = ax.plot(np.arange(row_length), np.zeros((row_length,1)), c='k', lw=3, label='row-wise intensity')
    row_peakPoints, = ax.plot(np.arange(row_length), np.zeros((row_length,1)), 'x', c='y', label='peaks')
    ax.set_title('Row wise intensity of segmented mask')
    ax.set_xlim(0, row_length)
    
    ax.set_ylim(0, 300)
    ax.legend()
    plt.ion()
    plt.show()


def get_liver_boundaries():
    """
    Returns : 1. reqd_boundary_points (Which is close to heart)
              2. reqd_boundary_rects  (Bounding box of above points)
              3. full_boundary_points (All the boundary points of liver)
    """
    print("Calculating liver boundaries")
    while True:
        (frame_grabbed, orig_frame) = vid_capture.read()
        (mask_grabbed, orig_mask) = msk_capture.read()
        
        if not mask_grabbed:
            break
        
        # Pre-process frame
        frame_gray = orig_frame
        
        mask_gray = cv2.cvtColor(orig_mask,cv2.COLOR_BGR2GRAY)
        
        ######### Start column sum #########
        mask_col_sum = np.sum((mask_gray/255),axis=0)
        
        # Find peaks
        peaks, _ = find_peaks(mask_col_sum, height=20, distance=5)
        
        # Plot data
        mask_col_sum = mask_col_sum.reshape(mask_col_sum.shape[0],1) 
        if(flg_vis_liv_boundary):    
            col_line.set_ydata(mask_col_sum)
            col_peakPoints.set_xdata(peaks)
            col_peakPoints.set_ydata(mask_col_sum[peaks])
        
        # Draw first col line
        first_col_peak = peaks[0]
        first_col_start_point = (first_col_peak-buffer_to_left, 0) 
        first_col_end_point = (first_col_peak-buffer_to_left, 600)
        if(flg_vis_liv_boundary):
            color = (255, 255, 0) 
            thickness = 1
            frame_gray = cv2.line(orig_frame, first_col_start_point, first_col_end_point, color, thickness) 
            
        # Draw second col line
        sec_col_start_point = (first_col_peak+buffer_to_right, 0) 
        sec_col_end_point = (first_col_peak+buffer_to_right, 600) 
        if(flg_vis_liv_boundary):
            color = (255, 255, 0) 
            thickness = 1
            frame_gray = cv2.line(frame_gray, sec_col_start_point, sec_col_end_point, color, thickness) 
    
        ######### Start row sum #########
        mask_row_sum = np.sum((mask_gray/255),axis=1)
        
        # Find peaks
        peaks, _ = find_peaks(mask_row_sum, height=0, distance=2)
        
        # Plot data
        mask_row_sum = mask_row_sum.reshape(mask_row_sum.shape[0],1)
        if(flg_vis_liv_boundary):
            row_line.set_ydata(mask_row_sum)
            row_peakPoints.set_xdata(peaks)
            row_peakPoints.set_ydata(mask_row_sum[peaks])
            
        # Draw first row line
        first_row_peak = peaks[0]
        first_row_start_point = (0, first_row_peak+buffer_to_up) 
        first_row_end_point = (800, first_row_peak+buffer_to_up) 
        if(flg_vis_liv_boundary):    
            color = (255, 255, 255) 
            thickness = 1
            frame_gray = cv2.line(frame_gray, first_row_start_point, first_row_end_point, color, thickness) 
        
        # Draw second row line which is the last peak
        last_row_peak = peaks[-1]
        sec_row_start_point = (0,last_row_peak-buffer_to_down)
        sec_row_end_point = (800, last_row_peak-buffer_to_down) 
        if(flg_vis_liv_boundary):    
            color = (255, 255, 255) 
            thickness = 1
            frame_gray = cv2.line(frame_gray, sec_row_start_point, sec_row_end_point, color, thickness) 
        
        # Draw bounding rectangle
        x = first_col_start_point[0]
        y = first_row_start_point[1]
    
        w = sec_col_start_point[0]-x
        h = sec_row_start_point[1]-y
        reqd_boundary_rects.append([(x,y),(x+w,y+h)])
        
        if(flg_vis_liv_boundary):    
            cv2.rectangle(frame_gray,(x,y),(x+w,y+h),(255,255,255),2)
        
        # Get full boundary from mask
        ret,thresh = cv2.threshold(mask_gray,200,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        tmp = np.zeros_like(frame_gray)
        full_boundary_matrix = cv2.drawContours(tmp, contours, -1, (255,255,255), 1)
        full_boundary_matrices.append(full_boundary_matrix)
        
        # Get full boundary coordinates
        full_boundary_indices = np.where(full_boundary_matrix == [255])
        full_boundary_points.append([full_boundary_indices[1],full_boundary_indices[0]])
        
        for point in zip(full_boundary_indices[1],full_boundary_indices[0]):
            cv2.circle(frame_gray,tuple(point),1,(0,0,255))        
    
        # Remove boundary points outside rectangle
        # Note: x and y coords are switched here, be careful, for above first x then y, for below, first y then x
        reqd_boundary = full_boundary_matrix.copy()
        reqd_boundary[:,0:x-3] = 0
        reqd_boundary[:,x+w-3:] = 0
        
        reqd_boundary[0:y-3,] = 0
        reqd_boundary[y+h-3:,] = 0
        
        # Get reqd boundary coordinates
        reqd_boundary_indices = np.where(reqd_boundary == [255])
        reqd_boundary_points.append([reqd_boundary_indices[1],reqd_boundary_indices[0]])    
        if(flg_vis_liv_boundary):    
            for point in zip(reqd_boundary_indices[1],reqd_boundary_indices[0]):
                cv2.circle(frame_gray,tuple(point),1,(255,0,255))
            
            cv2.imshow('Grayscale', frame_gray)
            fig.canvas.draw()
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid_capture.release()
    msk_capture.release()
    
    cv2.destroyAllWindows()
    

    
    return reqd_boundary_points, reqd_boundary_rects, full_boundary_points, full_boundary_matrices

# zz = get_liver_boundaries()















