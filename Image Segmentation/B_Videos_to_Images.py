# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:33:14 2021

@author: Khamar Uz Zama

1. Reads each video from each sub folder
2. Extracts key frames from each video
3. Saves them in ONE folder, with name formatted as "FolderName-VideoName-FrameID.jpg"

"""
import os
from os.path import isfile, join
from os import listdir
import numpy as np
import cv2
import math

import A_Config



def createKeyFrames():
    
    root = os.path.join(os.getcwd(),A_Config.videos_folder)
    folders = os.listdir(root)
    
    
    for folder in folders:
        videoFiles = [f for f in listdir(os.path.join(root,folder)) if isfile(join(root,folder, f))]
    
        print("Folder: ",folder)
        for videoFile in videoFiles:
            print("--Video:", videoFile)
            cap = cv2.VideoCapture(os.path.join(root,folder,videoFile))
            success,frame = cap.read()
            frameRate = cap.get(cv2.CAP_PROP_FPS)
            
            while(success):
              #ret, frame = cap.read()
              success,frame = cap.read()
    
              frameId = cap.get(1) #current frame number
        
              #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              #cv2.imshow('frame',gray)
              if(frameId % math.floor(frameRate) == 0):
    
                    filename = folder + "-" + videoFile[:-4] +  "-" + str(int(frameId)) + ".jpg"
                    cv2.imwrite(os.path.join(os.getcwd(),A_Config.frames_folder,filename), frame)
    return

createKeyFrames()