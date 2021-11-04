# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:59:37 2021

@author: Khamar Uz Zama

1. This program is not called by other programs. 
2. Flows are calculated individually in this program and saved separately in folder.

"""



import cv2
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
import pickle
import A_Config
import os
from os.path import join




#### Constants ######

videos_folder = A_Config.videos_folder
output_flow_folder = A_Config.output_flow_folder

video_file = A_Config.video_file 
mask_video_file = A_Config.mask_video_file
output_flows_file = A_Config.flows_file


def get_flows():
    bit_capture = cv2.VideoCapture(cv2.samples.findFile(join(os.getcwd(),videos_folder,video_file)))

    print("calculating flows for",video_file)
    ret, frame1 = bit_capture.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    flows = []
    i = 0
    while(1):
        ret, frame2 = bit_capture.read()
        if not ret:
            break
        
        frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prvs, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html
        # https://nanonets.com/blog/optical-flow/
        # In a high-level view, small motions are neglected as we go up the pyramid and large motions 
        # are reduced to small motions - we compute optical flow along with scale.
        # https://funvision.blogspot.com/2016/02/opencv-31-tutorial-optical-flow.html
        # https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
        # flow : computed flow image that has similar size as prev and type to be CV_32FC2.        
        # pyr_scale : parameter specifying the image scale to build pyramids for each image (scale < 1). A classic pyramid is of generally 0.5 scale, every new layer added, it is halved to the previous one.
        # levels : levels=1 says, there are no extra layers (only the initial image) . It is the number of pyramid layers including the first image.
        # winsize : It is the average window size, larger the size, the more robust the algorithm is to noise, and provide fast motion detection, though gives blurred motion fields.#
        # iterations : Number of iterations to be performed at each pyramid level.
        # poly_n : It is typically 5 or 7, it is the size of the pixel neighbourhood which is used to find polynomial expansion between the pixels.
        # poly_sigma : standard deviation of the gaussian that is for derivatives to be smooth as the basis of the polynomial expansion. It can be 1.1 for poly= 5 and 1.5 for poly= 7.
        # flags : It can be a combination of-
        #   OPTFLOW_USE_INITIAL_FLOW uses input flow as initial apporximation.
        #   OPTFLOW_FARNEBACK_GAUSSIAN uses gaussian winsize*winsize filter.
        
        flow = cv2.calcOpticalFlowFarneback(prev = prvs,
                                            next = frame2,
                                            flow = None, 
                                            pyr_scale = 0.5,
                                            levels = 5,
                                            winsize = 1000,
                                            iterations = 5,
                                            poly_n = 7,
                                            poly_sigma = 1.5,
                                            flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN )
        flows.append(flow)
        
        # Update previous frame every time
        # prvs = frame2
        
        # Update previous frame every xx frames
        if(not i%10):
            prvs = frame2

        i += 1
        #print(i)
        #cv2.imshow('frame2',vis)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    print("flows calculated")
    
    
    with open(join(os.getcwd(),output_flow_folder,output_flows_file), "wb") as fp:
        pickle.dump(flows, fp)
    
    return flows

flows = get_flows()











