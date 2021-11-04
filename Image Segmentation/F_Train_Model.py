# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:10:18 2020

@author: Khamar Uz Zama

@author: https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

This program:
1. Reads input data
2. Performs train test split of data
3. Trains the model
4. Visualized the results of training the model
5. Evaluates the model using validation data
6. Plots sample predictions

"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras import backend as K


import tensorflow as tf
import cv2

import A_Config
import D_Process_Data
import E_Create_Model


#### Constants ######
train_shape = A_Config.train_shape
train_results_folder = A_Config.train_results_folder
train_images_folder = A_Config.train_images_folder
train_masks_folder = A_Config.train_masks_folder

weights = A_Config.model_weights_file
model_name = A_Config.model_name
#####################


tf.config.experimental_run_functions_eagerly(True)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


border = 5
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1

my_preds = []
my_gd = []

temp_idx = 0

def tversky(y_true, y_pred):
    # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    smooth = 1
    alpha = 0.2
    # Lower alpha implies liver is weighted higher
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def dice_coef(y_true, y_pred):
    #y_pred = (y_pred > 0.5).astype(np.uint8)
    # my_preds.append(y_pred)
    # my_gd.append(y_true)    
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * (intersection + smooth)) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def my_dice_coef(y_true, y_pred):
    #dice_coeff_value = 0
    #my_pred = [tf.cast(pred, tf.int32, name=None) for pred in y_pred]
    rounded_preds = tf.math.round(y_pred)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(rounded_preds)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * (intersection + smooth)) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    #return dice_coeff_value

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def readInputData():

    image_names = next(os.walk(os.path.join(os.getcwd(),train_images_folder)))[2] # list of names all images in the given path
    mask_names = next(os.walk(os.path.join(os.getcwd(),train_masks_folder)))[2] # list of names all images in the given path
    
    if(len(image_names) != len(mask_names)):
        print("X != y")
        return
    
    print("No. of images = ", len(image_names))
    print("No. of masks = ", len(mask_names))

    
    X = np.zeros((len(image_names),train_shape[0], train_shape[1],train_shape[2] ), dtype=np.float32)
    y = np.zeros((len(image_names),train_shape[0], train_shape[1],train_shape[2]), dtype=np.float32)
    

    n = 0
    for image_name, mask_name in zip(image_names,mask_names):
        # print(image_name)
        # print(mask_name)
        # Load images
        img = load_img(train_images_folder+"/"+image_name, grayscale=True)
        x_img = img_to_array(img)
        #x_img = cv2.convertScaleAbs(x_img, 1.3, 1.3)
        #plt.imshow(x_img[...,0])
        #x_img = resize(x_img, (train_img_shape[0], A_Config.train_img_shape[1], A_Config.train_img_shape[2]), mode = 'constant', preserve_range = True)
        
        # Load masks
        #mask = load_img(os.path.join(os.getcwd(),train_masks_folder)+"\\"+id_[:-4]+"_mask.jpg", grayscale=True)
        mask = load_img(train_masks_folder+"/"+mask_name, grayscale=True)
        mask = img_to_array(mask)
        #mask = resize(mask, (train_masks_shape[0], A_Config.train_masks_shape[1], A_Config.train_masks_shape[2]), mode = 'constant', preserve_range = True)
        
        # Save images
        X[n] = x_img/255.0
        y[n] = mask/255.0
        n += 1
        
    return X,y

def visualizeInputData():

    # Visualize any randome image along with the mask
    ix = random.randint(0, len(X_train))
    has_mask = y_train[ix].max() > 0 # salt indicator
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))
    
    ax1.imshow(X_train[ix, ..., 0], cmap = 'seismic', interpolation = 'bilinear')
    if has_mask: # if salt
        # draw a boundary(contour) in the original image separating salt and non-salt areas
        ax1.contour(y_train[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
    ax1.set_title('Seismic')
    
    ax2.imshow(y_train[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
    ax2.set_title('Salt')
    
    return

def visualizeModelTraining(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss-tverszky-coeff")
    plt.plot(results.history["val_loss"], label="val-loss-tverskztverszky-coeff")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss - tverskz coeff")
    plt.legend()
    
    plt.savefig(os.path.join(os.getcwd(),train_results_folder,'Training loss.png'))
    
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve - Metric")
    plt.plot(results.history["dice"], label="Dice-coeff")
    plt.plot(results.history["val_dice"], label="val-Dice-coeff")
    plt.plot( np.argmax(results.history["val_dice"]), np.max(results.history["val_dice"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(os.path.join(os.getcwd(),train_results_folder,'Training accuracy.png'))

    
    
    return

def evaluateModel(model, X_valid, y_train):
    # load the best model
    model.load_weights(weights)
    
    # Evaluate on validation set (this must be equals to the best log_loss)
    model.evaluate(X_valid, y_valid, verbose=1)
    
    # Predict on train, val and test
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_valid, verbose=1)
    
    # Threshold predictions
    preds_train_t = (preds_train > 0.5)
    preds_val_t = (preds_val > 0.5)
    
    return preds_train, preds_val, preds_train_t, preds_val_t



def plot_sample(X, y, preds, binary_preds, ix=None):
    """Function to plot the results"""
    global temp_idx
    if ix is None:
        ix = random.randint(0, len(X)-1)

    #has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    #if has_mask:
        #ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('X')

    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    ax[1].set_title('y')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    # if has_mask:
    #     ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('y Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    # if has_mask:
    #     ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('y Predicted binary')
    
    fig.savefig(os.path.join(os.getcwd(),train_results_folder,str(temp_idx)+'Sample Results.png'))
    temp_idx += 1
    
    return

def trainModel(X,y):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)    
    
    #model = E_Create_Model.createModel(im_height, im_width, im_channels)
    model = E_Create_Model.createAnotherUnetModel(train_shape[0],train_shape[1],train_shape[2])
    
    #model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
        ModelCheckpoint(weights, verbose=1, save_best_only=True, save_weights_only=True)
    ]
    
    history = model.fit(X_train, y_train, batch_size=16, epochs=20, callbacks=callbacks,\
                        validation_data=(X_valid, y_valid))
        
    model.save(model_name)

    # history = model.fit(X_train, y_train, batch_size=32, epochs=100,
    #                     validation_data=(X_valid, y_valid))

    return model, history



X, y = readInputData()
X1,y1 = D_Process_Data.augmentNumpy(X,y)

X = np.vstack((X,X1))
y = np.vstack((y,y1))

X = np.vstack((X,X))
y = np.vstack((y,y))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)    

#visualizeInputData()

model, results = trainModel(X,y)
#model,results = trainModelUsingImageGenerator()

visualizeModelTraining(results)
preds_train, preds_val, preds_train_t, preds_val_t = evaluateModel(model, X_valid, y_valid)
plot_sample(X_train, y_train, preds_train, preds_train_t)








