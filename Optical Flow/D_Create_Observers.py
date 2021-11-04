# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:03:18 2021

@author: Khamar Uz Zama

This program creates 
1. Initial Observers 
2. Trajectories
3. Grid of Observers
"""

import numpy as np
import math 
import cv2
import A_Config
import os
import pickle

from os.path import join

x_no_of_patches = 3
y_no_of_patches = 3

size_of_patch = 15 # must be odd

pixelsPerCM = A_Config.pixelsPerCM
distance_from_liver_bd_x = A_Config.distance_from_liver_bd_x

initial_observer_angs_folder = A_Config.initial_observer_angs_folder
angs_file_initial_observers = A_Config.angs_file_initial_observers
mags_file_initial_observers = A_Config.mags_file_initial_observers
magnitude_threshold_intial_observers = A_Config.magnitude_threshold_intial_observers


x_buffer = 0
y_buffer = 0

def get_intersection_points(full_boundary_matrix,line):
    """
    # Finds the intersection between a line and the liver boundary 
    # and returns the  coordinates of intersection

    Parameters
    ----------
    full_boundary_matrix : Full boundary of the liver
    line : line as a four point array format - (x1,y1,x2,y2) 

    """
    # zLogicalAnd = np.logical_or(tmp_line,full_boundary_matrix)

    
    # lines = np.vstack([center[0], center[1], 700, center[1]]).T.reshape(-1, 2, 2)
    lines = np.vstack(line).T.reshape(-1, 2, 2)
    
    # Create temporary matrix and add the line to it
    tmp_line = np.zeros_like(full_boundary_matrix)
    tmp_line = cv2.polylines(tmp_line, lines, 0, (255,255,255))

    zLogicalAnd = np.logical_and(tmp_line,full_boundary_matrix)
    intersection_coords = np.where(zLogicalAnd == [True])
    
    return intersection_coords

def create_central_patch():
    # centers = create_vertical_patch_centers()
    x,y = A_Config.x, A_Config.y
    centers = [[x,y]]
    patches = []
    
    for center in centers:
        patchx = []
        patchy = []
        
        startX = center[0]-int(size_of_patch/2)
        startY = center[1]-int(size_of_patch/2)
        
        endX = center[0]+int(size_of_patch/2)
        endY = center[1]+int(size_of_patch/2)
        for y in range(startY, endY+1):
            for x in range(startX, endX+1):
                patchx.append(x)
                patchy.append(y)
        patch = (patchx,patchy)
        patches.append(patch)
        
    return centers,patches


def get_intersection_btw_line_and_liver(full_boundary_matrices,first_patch_center,x2,y2):
    
    line = [first_patch_center[0],first_patch_center[1],x2,y2]
    intersection_points = get_intersection_points(full_boundary_matrices[100],line)

    return intersection_points


def create_patches(centers):
    patches = []
    
    for center in centers:
        patchx = []
        patchy = []
        
        startX = int(center[0]-int(size_of_patch/2))
        startY = int(center[1]-int(size_of_patch/2))
        
        endX = int(center[0]+int(size_of_patch/2))
        endY = int(center[1]+int(size_of_patch/2))
        for y in range(startY, endY+1):
            for x in range(startX, endX+1):
                patchx.append(x)
                patchy.append(y)
        patch = (patchx,patchy)
        patches.append(patch)
        
    return patches

def create_horizontal_patches(reqd_boundary_points,full_boundary_matrices,first_patch_center, angle_of_movement):
    """
    

    Parameters
    ----------
    reqd_boundary_points : Boundary points which are closer to the heart. We use this to create the first patch
    full_boundary_matrices : All the boundary points of liver in matrix form. Used to calculate intersection of line and boundary

    Returns
    -------
    None.

    """

    """
    Function
    2. Add line in the direction of movement from the first patch
    3. Find intersection between line and liver using the boundary
    4. Create patch centers on the line
    5. Add patches
    """
    centers = []

    # 2. Add line in the direction of movement from the first patch
    slope = math.tan(angle_of_movement)
    x2 = 800
    y2 = np.int32(slope *(x2-first_patch_center[0])+first_patch_center[1])
    
    # 3. Find intersection between line and liver using the boundary    
    intersection_points = get_intersection_btw_line_and_liver(full_boundary_matrices,first_patch_center,x2,y2)
    
    
    x1,y1 = int(first_patch_center[0]), int(first_patch_center[1])
    x2,y2 = int(intersection_points[1][0]), int(intersection_points[0][0])

    # 4. Find Max distance between first patch and the intersection point
    max_distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    max_distance = max_distance - 5
    
    # Append first center
    centers.append([x1,y1])
    center_count = 0
    while True:
        x2 = centers[center_count][0] - math.cos(angle_of_movement)*pixelsPerCM
        y2 = centers[center_count][1] - math.sin(angle_of_movement)*pixelsPerCM
        
        tot_distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
        
        if(tot_distance>max_distance):
            break
        else:
            centers.append([int(x2),int(y2)])
        
        center_count += 1
    
    # Append intersection point
    # x2,y2 = int(intersection_points[1][0]), int(intersection_points[0][0])    
    # centers.append([x2,y2])

        
    patches = create_patches(centers)

    
    return centers, patches

def create_first_patch_center(reqd_boundary_point):
    y = min(reqd_boundary_point[1]) + size_of_patch
    
    x = reqd_boundary_point[0][0] + distance_from_liver_bd_x
    
    return (x,y)

def get_next_patch(reqd_boundary_point,current_patch):
    _, current_patch_y = current_patch[0], current_patch[1]
    next_patch_y  = current_patch_y  + A_Config.pixelsPerCM + 0 # Increase 0 to increase distance between rows of grid
    if(next_patch_y > max(reqd_boundary_point[1])):
        return False,False

    y_indices = list(np.where(reqd_boundary_point[1] == next_patch_y))


    x_es = [reqd_boundary_point[0][y] for y in y_indices]
    
    next_patch_x = min(x_es)[0] + distance_from_liver_bd_x
    return next_patch_x, next_patch_y

def create_vertical_patch_centers(reqd_boundary_point,first_patch_center):
    vertical_centers = []
    
    max_distance = max(reqd_boundary_point[1]) + min(reqd_boundary_point[1])
    current_patch = first_patch_center
    vertical_centers.append([int(first_patch_center[0]),int(first_patch_center[1])])
    
    while(True):
        next_patch_x, next_patch_y  = get_next_patch(reqd_boundary_point,current_patch)
        if(next_patch_x):
            current_patch = (next_patch_x, next_patch_y)
            vertical_centers.append([int(next_patch_x),int(next_patch_y)])
        else:
            break
        
        # if((next_patch_y-first_patch_center[1]) > max_distance):
        #     break
        # else:
        #     current_patch = (next_patch_x, next_patch_y)
        #     vertical_centers.append([int(next_patch_x),int(next_patch_y)])
            
        
    return vertical_centers

def read_angles_of_init_observers():

    with open(join(os.getcwd(),initial_observer_angs_folder,angs_file_initial_observers), "rb") as fp:
        angs = pickle.load(fp)
    with open(join(os.getcwd(),initial_observer_angs_folder,mags_file_initial_observers), "rb") as fp:
        mags = pickle.load(fp)
        
    nr_of_observers = len(angs[0,:])
    init_observer_angles = [0]*len(angs[0])
    for i in range(0, nr_of_observers):
        threshold_indices = np.where(mags[:,i] > magnitude_threshold_intial_observers)
        threshold_angs = angs[:,:][threshold_indices]
        if(np.median(threshold_angs[:,i]) < 90):
           init_observer_angles[i] = np.median(threshold_angs[:,i]) + 180
        else:
            init_observer_angles[i] = np.median(threshold_angs[:,i])
            
    return init_observer_angles


def create_grid(reqd_boundary_points,full_boundary_matrices,vertical_only=False):
    
    reqd_boundary_point = reqd_boundary_points[24]
    
    # 1. Create first patch center using boundary
    first_patch_center = create_first_patch_center(reqd_boundary_point)
    
    # 2. Create vertical centers possible along the heart boundary
    vertical_centers = create_vertical_patch_centers(reqd_boundary_point,first_patch_center)
    
    if(vertical_only):
        vertical_patches = create_patches(vertical_centers)
        return vertical_centers, vertical_patches

    init_observer_angles = read_angles_of_init_observers()
    
    # vertical_centers = [[first_patch_center[0],first_patch_center[1]]]

    grid_width = 15
    grid_height = 15
    grid_matrix_centers = [[0 for x in range(grid_width)] for y in range(grid_height)]
    grid_matrix_patches = [[0 for x in range(grid_width)] for y in range(grid_height)] 

    
    for row_id,vertical_center in enumerate(vertical_centers):
        angle_of_movement = init_observer_angles[row_id]
        # angle_of_movement = A_Config.angle_of_movement # in degrees
        print("In degrees",angle_of_movement)
        angle_of_movement = math.radians(angle_of_movement) # in radians - required for math functions
        centers,patches = create_horizontal_patches(reqd_boundary_points,full_boundary_matrices,(vertical_center[0],vertical_center[1]),angle_of_movement)
        col_id = 0
        for center,patch in zip(centers,patches):
            grid_matrix_centers[row_id][col_id] = center
            grid_matrix_patches[row_id][col_id] = patch
            col_id += 1
    
    grid_matrix_centers = np.array(grid_matrix_centers,dtype = object)
    grid_matrix_patches = np.array(grid_matrix_patches,dtype = object)
    
    # Remove rows and columns with all zeros
    grid_matrix_centers = grid_matrix_centers[~np.all(grid_matrix_centers == 0, axis=1)]
    grid_matrix_centers = grid_matrix_centers[:, ~np.all(grid_matrix_centers == 0, axis=0)]
    grid_matrix_patches = grid_matrix_patches[~np.all(grid_matrix_patches == 0, axis=1)]
    grid_matrix_patches = grid_matrix_patches[:, ~np.all(grid_matrix_patches == 0, axis=0)]
    
    return grid_matrix_centers, grid_matrix_patches