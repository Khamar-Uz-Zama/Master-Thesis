# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:19:21 2020

@author: Khamar Uz Zama

1. Reads the input images and masks
2. Crops the images and masks
3. Adds filters to images
4. Saves them in folders

Note: The ouput after cropping are saved in temp folder addFilters() uses the images
from that folder

Input: 1. Images from 0_Data_frames folder
       2. Masks from 0_Data_Masks folder
       
Output: 1. Cropped+Resized+Filtered  Images are saved in 1_Train_Images folder
        2. Cropped+Resized Masks are saved in 2_Train_Masks folder       

Current Status of dataset:
    ## Created before training ##
    1. One set from from PreProcessImagesAndMasks
    2. Three sets from DataGenerator-3PreProcessingImages
        2.1 Original - without any augmentation
        2.2 Grid Distorted
        2.3 HorizontalFlip
        
    ## Created while training ##    
    3. One set from ImgAug def augmentNumpy(images, masks)

"""


import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import imgaug as ia
import imgaug.augmenters as iaa
from random import randint

import A_Config

#### Constants ######
orig_frames_folder = A_Config.frames_folder
orig_masks_folder = A_Config.masks_folder

train_temp_folder = A_Config.train_temp_folder

train_images_folder = A_Config.train_images_folder
train_masks_folder = A_Config.train_masks_folder

img_mask_dims = A_Config.train_shape


#### Crop dims ######
y1 = A_Config.y1
y2 = A_Config.y2
x1 = A_Config.x1
x2 = A_Config.x2

#####################

def throwError(image_name):
    print("Error... Image not found", image_name)
    raise Exception("Image not found")


def readFiles(folder, extension="jpg"):

    files = glob.glob(folder+"/*."+extension)
    X_data = []

    for myFile in files:
        image = cv2.imread (myFile)
        X_data.append (image)

    print('Files shape:', np.array(X_data).shape)
    return np.array(X_data)

# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def CropAndResizeImages(frames_folder, temp_folder, img_mask_dims):
    
    images_names = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]

    print("--Cropping and resizing Images")
    
    # Save cropped images in 99_Temp folder
    for image_name in images_names:
        img = cv2.imread(os.path.join(os.getcwd(),frames_folder,image_name),cv2.IMREAD_GRAYSCALE)
        if(img is None):
            print("Error reading Images")
            throwError(image_name)
            return
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (img_mask_dims[0],img_mask_dims[1]), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(os.getcwd(),temp_folder,image_name),img)
    print("-----Final images Shape = ", img.shape)
    print("Cropped and resized Images saved in folder:",temp_folder)
    return 1

def CropAndResizeMasks(masks_folder, output_folder, img_mask_dims):
        
    masks_names = [img for img in os.listdir(masks_folder) if img.endswith(".jpg")]

    print("--Cropping and resizing Masks")
    # Save cropped masks in MASKS folder
    for mask_name in masks_names:
        msk = cv2.imread(os.path.join(os.getcwd(),masks_folder,mask_name),cv2.IMREAD_GRAYSCALE)
        if(msk is None):
            print("Error cropping Masks")
            throwError(mask_name)
            return
        
        msk = msk[y1:y2, x1:x2]        
        msk = cv2.resize(msk, (img_mask_dims[0],img_mask_dims[1]), interpolation=cv2.INTER_LINEAR)
        
        name = mask_name.split("/")[-1].split(".")
        temp_msk_name = name[0]
        msk_name = temp_msk_name+"_0."+name[1]
        
        cv2.imwrite(os.path.join(os.getcwd(),output_folder,msk_name),msk)
        
    print("-----Final Mask Shape = ", msk.shape)
    print("Cropped masks saved in folder:",output_folder)
    
    return 1

def addFiltersToImages(temp_folder,output_folder):
    
    print("--Adding Filters to images")
    inc_contrast = True
    gauss_blur = False
    bilateral_filt = False
    canny=False

    images_names = [img for img in os.listdir(temp_folder) if img.endswith(".jpg")]
        
    for image_name in images_names:
        img = cv2.imread(os.path.join(os.getcwd(),temp_folder,image_name),cv2.IMREAD_GRAYSCALE)
        
        if(img is None):
            print("Error adding filters to images")
            throwError(image_name)
            return None
        if(inc_contrast):
            img = cv2.equalizeHist(img)
        if(gauss_blur):
            img = cv2.GaussianBlur(img,(5,5),0)
        if(bilateral_filt):
            img = cv2.bilateralFilter(img, d=10, sigmaColor=80, sigmaSpace=80)
        if(canny):
            img = cv2.Canny(img, threshold1=100, threshold2=200, apertureSize =3)
        
        name = image_name.split("/")[-1].split(".")
        
        temp_img_name = name[0]
        img_name = temp_img_name+"_0."+name[1]
        cv2.imwrite(os.path.join(os.getcwd(),output_folder,img_name),img)   
            
    print("-----Final images shape = ", img.shape)
    print("Filtered images saved in folder:",output_folder)

    return 1

def PreProcessImagesAndMasks(frames_folder,masks_folder):
    # images = readFiles(orig_frames_folder)
    # masks = readFiles(orig_masks_folder)
    
    
    status = CropAndResizeImages(frames_folder,train_temp_folder,img_mask_dims)
    if(status is None):
        print("Error while cropping images and masks")
        return
    
    status = CropAndResizeMasks(orig_masks_folder,train_masks_folder,img_mask_dims)
    if(status is None):
        print("Error while cropping images and masks")
        return    
    
    status = addFiltersToImages(train_temp_folder,train_images_folder)
    if(status is None):
        print("Error while adding filters to masks")
        return Exception()
    
    return 1

# This is not called while training. The files are augmented before itself and saved
# PreProcessImagesAndMasks(orig_frames_folder,orig_masks_folder)

# This is called while training. The files are augmented dynamically and not stored
def augmentNumpy(images, masks):
    """
    Images should be numpy arrays
    """
    images = np.asarray(images)
    masks = np.asarray(masks)
    
    # seed = randint(0,10)
    # np.random.bit_generator = np.random._bit_generator

    # ia.seed(seed)
    
    augmenters_imgs = iaa.Sequential([
        iaa.Dropout([0.05, 0.1]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 0.8)),       # sharpen the image
        iaa.Affine(rotate=(-25, 25)),  # rotate by -10 to 10 degrees (affects segmaps)
        #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=False)           
    # ia.seed(seed)

    augmenters_msks = iaa.Sequential([
        iaa.Sharpen((0.0, 0.8)),       # sharpen the image
        iaa.Affine(rotate=(-25, 25)),  # rotate by -10 to 10 degrees (affects segmaps)
        #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=False)                 
    
    seq_imgs = iaa.Sequential(augmenters_imgs, random_order=False)
    seq_msks = iaa.Sequential(augmenters_msks, random_order=False)        
        
    seq_imgs_deterministic = seq_imgs.to_deterministic()
    seq_msks_deterministic = seq_msks.to_deterministic()

    imgs_aug = seq_imgs_deterministic.augment_images(images)
    masks_aug = seq_msks_deterministic.augment_images(masks)
    
    return imgs_aug, masks_aug


    