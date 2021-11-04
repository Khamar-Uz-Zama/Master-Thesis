# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 23:42:57 2021

@author: Khamar Uz Zama
"""

#### Folders ######

videos_folder = "0_Data"
frames_folder = "0_Data_Frames"
masks_folder = "0_Data_Masks"

train_images_folder = "1_Train_Images"
train_masks_folder = "2_Train_Masks"
train_results_folder = "3_Training_Results"

test_images_folder = "4_Test_Images"

pred_masks_folder = "5_Predicted_Masks"
pred_video_folder = "6_Predicted_Video"

train_temp_folder = "99_Train_Temp"
test_temp_folder = "99_Test_Temp"
test_frames_temp_folder = "99_Test_Frames_Temp"

test_overlays_folder = "99_Test_Overlays"
#### Files ######

annotations_file = "A_a_Labels.csv"

# test_video_file = "A_a_Test_Video1.wmv"
# test_video_file = "A_a_Test_Video2.mp4"

test_video_file = "Run2yy5.mp4"
# test_video_files = ["Run12.mp4","Run13.mp4","Run11.mp4"]
test_video_files = ["INKA015-Run14.mp4"]



# output files 
predicted_video_file = "Predicted.avi"
overlayed_video_file = "Overlayed.avi"
bitwise_and_video_file = "bitwise.avi"

#### Shapes #####

orig_masks_shape = (600,800,3) # Change later, use default image shape

train_shape = (128, 128, 1)
test_shape = (128, 128, 1)


#### Crop dimensions #####
# 1. These dimensions are used  to crop the unnecessary (from all sides) of the image
# 2. These are again used to resize the mask to overlay on original video
y1 = 100
y2 = 580
x1 = 100
x2 = 720



#### Model #####

use_preTrained_model = False
model_name = "FullModel"

model_weights_file = "model-dice.h5"

# assert imshape[0]%32 == 0 and imshape[1]%32 == 0,\
#     "imshape should be multiples of 32. comment out to test different imshapes."