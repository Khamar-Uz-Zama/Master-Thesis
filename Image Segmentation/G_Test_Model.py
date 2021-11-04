# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 21:57:45 2020

@author: Khamar Uz Zama


This program:
1. Reads the video for testing 
2. Splits the video into images and predicts the liver regions
3. Saves the predictions into separate folder
4. Creates a video from the predictions and saves the BITWISE_AND video

"""
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import A_Config
import D_Process_Data
import E_Create_Model

#### Constants ######
train_shape = A_Config.train_shape

img_mask_dims = A_Config.train_shape

frames_folder = A_Config.test_frames_temp_folder
temp_folder = A_Config.test_temp_folder
test_images_folder = A_Config.test_images_folder
test_overlays_folder = A_Config.test_overlays_folder

pred_masks_folder = A_Config.pred_masks_folder
pred_video_folder = A_Config.pred_video_folder

input_videoFile = A_Config.test_video_file
output_video_file = A_Config.predicted_video_file
output_overlay_video_file = A_Config.overlayed_video_file
output_bitwise_and_video_file = A_Config.bitwise_and_video_file
weights = A_Config.model_weights_file

#### Crop dims ######
y1 = A_Config.y1
y2 = A_Config.y2
x1 = A_Config.x1
x2 = A_Config.x2

#### Temp vars ######
temp_idx = 0

def delete_all_files_in_folder():
    """
    Deltes all the images in the folders
    """
    
    folders_to_delete = [frames_folder,temp_folder, test_images_folder, test_overlays_folder, pred_masks_folder]
    
    for folder_to_delete in folders_to_delete:
        print("deleting files in ",folder_to_delete)
        file_names = [img for img in os.listdir(folder_to_delete) if img.endswith(".jpg")]
        
        for file_name in file_names:
            os.remove(os.path.join(os.getcwd(),folder_to_delete,file_name))
    
    print("old files deleted")
    
    return



def extract_frames_from_video(frames_folder,input_videoFile):
    """
    Save images in 99_Test_Frames_Temp
    """
    cap = cv2.VideoCapture(input_videoFile)
    frameRate = cap.get(5) #frame rate
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        # if (frameId % math.floor(frameRate) == 0):
        x_img = img_to_array(frame)
        image_name = str(int(frameId))+"TrueFrame.jpg"
        cv2.imwrite(os.path.join(os.getcwd(),frames_folder,image_name),x_img)
        
    cap.release()
    
    return 

def get_predictions_for_frames(model, test_images_folder):
    """
    Loads the test images from a folder and returns the predictions
    """    
    images_names = [img for img in os.listdir(test_images_folder) if img.endswith(".jpg")]
    images_names = sorted(images_names, key=numericalSort)


    images = []
    for img_name in images_names:
        img = cv2.imread(os.path.join(os.getcwd(),test_images_folder,img_name),cv2.IMREAD_GRAYSCALE)    
        img = img/255.0

        images.append(img)
        
    images = np.asarray(images)        
    print("Predicting masks")
    preds_mask = model.predict(images, verbose=1)

    return preds_mask, images

def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')

    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    
    return parts


def create_video_from_masks(pred_masks_folder, pred_video_folder, output_video_file):
    """
    takes prediction masks and converts them to video
    """
    
    images = [img for img in os.listdir(pred_masks_folder) if img.endswith(".jpg")]
    images = sorted(images, key=numericalSort)
    frame = cv2.imread(os.path.join(os.getcwd(),pred_masks_folder, images[0]))
    height, width, layers = frame.shape
  
    video = cv2.VideoWriter(os.path.join(os.getcwd(),pred_video_folder,input_videoFile[:-4]+"_"+output_video_file), 0, 29, (width,height))
    
    for image in images:
        img =  cv2.imread(os.path.join(os.getcwd(),pred_masks_folder, image))
        video.write(img)
    
    cv2.destroyAllWindows()
    video.release()

    return 


def load_model():  
    """
    Loads the model and returns it
    """
    # model = E_Create_Model.createAnotherUnetModel()
    model = E_Create_Model.createAnotherUnetModel(train_shape[0],train_shape[1],train_shape[2])

    print("Loading model")

    # load the best model
    model.load_weights(weights)
    return model

def save_pred_masks(preds,pred_masks_folder):
    import matplotlib
    
    for i,pred in enumerate(preds):  
        pred = pred.astype(np.float32)
        
        # Resize from 128x128 to 480x620(heightxwidth)
        pred = cv2.resize(pred, (620,480), interpolation=cv2.INTER_LINEAR) # opposite order here
        
        # Refill with 0 from 480x620 to 600x800(heightxwidth) using crop dims
        pred = cv2.copyMakeBorder(pred,100,20,90,90,cv2.BORDER_CONSTANT,value=0) # (top,bottom,left,right)

        pred = pred.astype(np.uint8)
        plt.imsave(os.path.join(pred_masks_folder, str(i)+'pred_masks.jpg'),pred,cmap=matplotlib.cm.gray)
        
    return

def plot_sample(X, preds, binary_preds, ix=None):
    """Function to plot the results"""
    global temp_idx
    if ix is None:
        ix = random.randint(0, len(X)-1)

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X[ix], cmap='gray')
    ax[0].set_title('X')

    ax[1].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    ax[1].set_title('y Predicted')
    
    ax[2].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    ax[2].set_title('y Predicted binary')
    
    fig.savefig(os.path.join(os.getcwd(),pred_video_folder,str(temp_idx)+'Predictions.png'))
    temp_idx += 1
    
    return

def create_overlay_video_from_masks(pred_masks_b, frames_folder):
    """
    Loads the test images from frames folder, processes them in right format
    Takes the predictions and rescales them back to original size and overlays them on true images,
    Overlayed images are saved in the folder and they are converted to video using another function call
    """
    
    print("Overlaying masks on images")
    
    images_names = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
    images_names = sorted(images_names, key=numericalSort)
    
    processed_masks = []
    
    idx = 0    
    # Load test images from frames folder
    for image_name in images_names:
        img = cv2.imread(os.path.join(os.getcwd(),frames_folder,image_name))
        img = np.asarray(img)
        img = img.astype(np.float64)
        img = img/255
    
        #true_frames.append(img)
        
        pred_mask = pred_masks_b[idx]
        pred_mask = pred_mask.astype(np.float32)
        
        # Resize from 128x128 to 480x620(heightxwidth)
        pred_mask = cv2.resize(pred_mask, (620,480), interpolation=cv2.INTER_LINEAR) # opposite order here
        
        # Refill with 0 from 480x620 to 600x800(heightxwidth) using crop dims
        pred_mask = cv2.copyMakeBorder(pred_mask,100,20,90,90,cv2.BORDER_CONSTANT,value=0) # (top,bottom,left,right)
        
        # Convert back to 3d
        pred_mask = cv2.cvtColor(pred_mask,cv2.COLOR_GRAY2RGB)
        pred_mask = pred_mask.astype(np.float64)
    
        # processed_masks.append(pred_mask)
        
        idx += 1
        
        fin = cv2.addWeighted(pred_mask, 0.8, img, 1.0, 0)
    
        
        image_name = str(idx)+'-TestOverlay.jpg'
        save_img(os.path.join(os.getcwd(),test_overlays_folder,image_name), fin)
    
        
    print("Overlaying successfull on images", len(images_names))
    
    print("Converting to overlay video")
    create_video_from_masks(test_overlays_folder, pred_video_folder, output_overlay_video_file)
    print("Overlay video saved as: ",input_videoFile[:-4]+"-"+output_overlay_video_file)

    return


def create_bitwise_and_video_from_masks(pred_masks_b, frames_folder):
    images_names = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
    images_names = sorted(images_names, key=numericalSort)
        
    idx = 0    
    # Load test images from frames folder
    for image_name in images_names:
        img = cv2.imread(os.path.join(os.getcwd(),frames_folder,image_name))
        img = np.asarray(img)
        img = img.astype(np.float64)
        img = img/255
    
        #true_frames.append(img)
        
        pred_mask_b = pred_masks_b[idx]
        pred_mask_b = pred_mask_b.astype(np.float32)
        
        # Resize from 128x128 to 480x620(heightxwidth)
        pred_mask_b = cv2.resize(pred_mask_b, (620,480), interpolation=cv2.INTER_LINEAR) # opposite order here
        
        # Refill with 0 from 480x620 to 600x800(heightxwidth) using crop dims
        pred_mask_b = cv2.copyMakeBorder(pred_mask_b,100,20,90,90,cv2.BORDER_CONSTANT,value=0) # (top,bottom,left,right)
        pred_mask_b = (pred_mask_b > 0.5)
        pred_mask_b = pred_mask_b.astype(np.float32)

        # Convert back to 3d
        #pred_mask_b = cv2.cvtColor(pred_mask_b,cv2.COLOR_GRAY2RGB)
        pred_mask_b = pred_mask_b.astype(np.uint8)
        
        fin = cv2.bitwise_and(img,img,mask=pred_mask_b)
        
        idx += 1    
        image_name = str(idx)+'-TestOverlay.jpg'
        save_img(os.path.join(os.getcwd(),test_overlays_folder,image_name), fin)
    
        
    print("Performing BITWISE_AND on images", len(images_names))
    
    print("Converting to video")
    create_video_from_masks(test_overlays_folder, pred_video_folder, output_bitwise_and_video_file)
    print("Video saved as: ",input_videoFile[:-4]+"-"+output_bitwise_and_video_file)    
    
    return


print("--Testing model--")


# delete_all_files_in_folder()
# model = load_model()
# extract_frames_from_video(frames_folder,input_videoFile)
# D_Process_Data.CropAndResizeImages(frames_folder, temp_folder, img_mask_dims)
# D_Process_Data.addFiltersToImages(temp_folder, test_images_folder)

# pred_masks, frames = get_predictions_for_frames(model, test_images_folder)

# pred_masks_b = (pred_masks > 0.5)
# pred_masks = pred_masks_b

# #plot_sample(frames, pred_masks, pred_masks_b)

# save_pred_masks(pred_masks,pred_masks_folder)

# vid = create_video_from_masks(pred_masks_folder, pred_video_folder, output_video_file)

# # create_overlay_video_from_masks(pred_masks_b, frames_folder)

# create_bitwise_and_video_from_masks(pred_masks_b, frames_folder)

model = load_model()

for input_videoFile in A_Config.test_video_files:    
    delete_all_files_in_folder()
    
    extract_frames_from_video(frames_folder,input_videoFile)
    D_Process_Data.CropAndResizeImages(frames_folder, temp_folder, img_mask_dims)
    D_Process_Data.addFiltersToImages(temp_folder, test_images_folder)
    
    pred_masks, frames = get_predictions_for_frames(model, test_images_folder)
    
    pred_masks_b = (pred_masks > 0.5)
    pred_masks = pred_masks_b
    
    #plot_sample(frames, pred_masks, pred_masks_b)
    
    save_pred_masks(pred_masks,pred_masks_folder)
    
    vid = create_video_from_masks(pred_masks_folder, pred_video_folder, output_video_file)
    
    # create_overlay_video_from_masks(pred_masks_b, frames_folder)
    
    create_bitwise_and_video_from_masks(pred_masks_b, frames_folder)














































####
# Function to test tversky and alpha parameter.... working code but not useful for now
# def Test_Tversky(y_true, y_pred):
#     y_true = y_true[...,0]
#     y_pred = tf.cast(y_pred, tf.float32)

#     # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
#     smooth = 1
#     alpha = 0.7
#     # Lower alpha implies liver is weighted higher
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#     false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    
#     return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
# sc = Test_Tversky(y_true, y_pred)#
# print("loss=",sc.numpy())
####


#### Below code is overlaying predicted mask on image as PLOT.. working but not useful for now....
# idx = 0
# for mask,img in zip(pred_masks,frames):
#     ix = random.randint(0, len(pred_masks)-1)
#     mask, img = pred_masks[ix,...,0], frames[ix]
#     masked = np.ma.masked_where(mask == 0, mask)
    
#     plt.figure()
#     # plt.subplot(1,2,1)
#     # plt.imshow(img, 'gray', interpolation='none')
#     # plt.subplot(1,2,1)
    
#     plt.imshow(img, 'gray', interpolation='none')
#     plt.imshow(masked, 'jet', interpolation='none', alpha=0.7)
#     #plt.show()
    
#     plt.savefig(os.path.join(os.getcwd(),test_overlays_folder,str(idx)+'-Test Overlay.png'))
#     idx += 1
    
#     if(idx == 10):
#         return
####        