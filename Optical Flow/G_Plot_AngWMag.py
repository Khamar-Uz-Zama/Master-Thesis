# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:11:47 2021

@author: Khamar Uz Zama

This program:
    1. Reads flows and angs
    2. Detects peaks of angs and extracts the data in the window size
    3. Creates the subplots of angs mags/flows
    4. Creates box plot of all the mags
    5. Extract medians from the box plots and saves them (Used in multi-liver plots)
    6. Calculates the damping values of Magnitude's median
    
"""

import pickle
import A_Config
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from os.path import join


# BGR
patch_colors_BGR = [(0, 0, 255), # Red
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

# Convert to RGB
patch_colors_RGB = [(color[2],color[1],color[0]) for color in patch_colors_BGR]    
patch_colors_RGB = np.array(patch_colors_RGB,float) / 255

video_file = A_Config.video_file
angs_file_initial_observers = A_Config.angs_file_initial_observers
patch_flows_file = A_Config.patch_flows_file
peak_stats_file = A_Config.peak_stats_file
magnitude_threshold_intial_observers = A_Config.magnitude_threshold_intial_observers
input_individual_patch_flows_folder = A_Config.individual_patch_flows_folder
initial_observer_angs_folder = A_Config.initial_observer_angs_folder
output_all_peak_stats_folder =  A_Config.all_peak_stats_folder
output_all_peak_results_folder = A_Config.all_peak_results_folder
this_video_results_folder = video_file[0:-4]

if not os.path.isdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder)):
    os.mkdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder))
# else:
#     filelist = [ f for f in os.listdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder))]
#     for f in filelist:
#         os.remove(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder, f))

def read_data():
    """
    Read Individual flows and angles of observers to sub plot the data
    """
    with open(join(os.getcwd(),input_individual_patch_flows_folder,patch_flows_file), "rb") as fp:
        patch_flows = pickle.load(fp)
        
    with open(join(os.getcwd(),initial_observer_angs_folder,angs_file_initial_observers), "rb") as fp:
        angs = pickle.load(fp)
        
    return patch_flows, angs


def patch_artist(bp, ax_box_plot):

    ax_box_plot.title.set_text('Distance vs Magnitude of '+video_file[:-4])
    ax_box_plot.set_ylabel('Magnitude')
    ax_box_plot.set_xlabel('Distance from initial observer')

    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)    




def plot_box_plot_of_col_wise_mags(all_col_wise_peak_values):
    all_col_wise_peak_values = np.array(all_col_wise_peak_values,dtype=object,)
    fig_box_plot = plt.figure(1, figsize=(9, 6))
    ax_box_plot = fig_box_plot.add_subplot(111)
    ax_box_plot.set_title("Box plot of " +video_file[:-4])
    
    bp = ax_box_plot.boxplot(all_col_wise_peak_values,positions=np.arange(0,len(all_cols_all_values)).tolist(),patch_artist = True)
    patch_artist(bp,ax_box_plot)
    
    plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"_BoxPlot.png"
    box_plot_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)
    plt.savefig(box_plot_path)
    fig_box_plot.clf()
    plt.clf()
    
    return

def plot_damping_values_of_magnitude(all_rowwise_threshold_values, stat):

    
    fig = plt.figure()
    ax0 = fig.add_subplot(111)

    ax0.set_ylabel('% Decrease in magnitudes')
    ax0.set_xlabel('Distance from initial observer')
    
    lines = [None]*len(all_rowwise_threshold_values)
    nr_of_rows = len(all_rowwise_threshold_values[:,0])
    legends = []
    if(stat == 0):
        save_title = "Mean.png"
        ax0.title.set_text('Damping of mean of magnitudes for '+video_file[:-4])
        for this_row in range(0, nr_of_rows):
            threshold_values = all_rowwise_threshold_values[this_row,:]
            damping_values = []
            for value in threshold_values:
                if value is not None:
                    damping_values.append((np.mean(value)/np.mean(threshold_values[0]))*100)
            x = np.arange(len(damping_values))
            lines[this_row], = ax0.plot(x, damping_values, color=patch_colors_RGB[this_row])
            legends.append("Trajectory "+str(this_row+1))
        ax0.legend(lines, (legends), loc='upper right')
        
    elif(stat == 1):
        save_title = "Median.png"
        ax0.title.set_text('Damping of median of magnitudes for '+video_file[:-4])
        for this_row in range(0, nr_of_rows):
            threshold_values = all_rowwise_threshold_values[this_row,:]
            damping_values = []
            for value in threshold_values:
                if value is not None:
                    damping_values.append((np.median(value)/np.median(threshold_values[0]))*100)
            x = np.arange(len(damping_values))
            lines[this_row], = ax0.plot(x, damping_values, color=patch_colors_RGB[this_row])
            legends.append("Trajectory "+str(this_row+1))
        ax0.legend(lines, (legends), loc='upper right')        
    elif(stat == 2):
        save_title = "Max.png"
        ax0.title.set_text('Damping of max of magnitudes for '+video_file[:-4])
        for this_row in range(0, nr_of_rows):
            threshold_values = all_rowwise_threshold_values[this_row,:]
            damping_values = []
            if(len(threshold_values)>0):
                for value in threshold_values:
                    if value is not None:
                        try:
                            damping_values.append((np.max(value)/np.max(threshold_values[0]))*100)
                        except ValueError:
                            print("value error")                            
                            pass
            else:
                print("length of threshold values is zero")
            x = np.arange(len(damping_values))
            lines[this_row], = ax0.plot(x, damping_values, color=patch_colors_RGB[this_row])
            legends.append("Trajectory "+str(this_row+1))
        ax0.legend(lines, (legends), loc='upper right')
    

        
    plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p_")+save_title
    plot_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)
    plt.savefig(plot_path)
    
    return



def compress_data_at_patch_level(patch_flows):
    compressed_patch_flows = []
    for frame_count,patch_flow in enumerate(patch_flows):
        nr_of_rows = len(patch_flow)
        nr_of_cols = len(patch_flow[0,:])
        for i in range(0,nr_of_rows):
            for j in range(0,nr_of_cols):
                if patch_flow[i,j] is not None:
                    patch_flow[i,j] = np.mean(patch_flow[i,j])
        compressed_patch_flows.append(patch_flow)
        
    return compressed_patch_flows

def transform_data_for_box_plot(patch_flows,magnitude_threshold_intial_observers):
    nr_of_cols = len(patch_flows[0][0,:])
    all_cols_all_values = []
    for this_col in range(0, nr_of_cols):
        this_col_threshold_values = []
        for frame_count,patch_flow in enumerate(compressed_patch_flows):
            this_col_flows = patch_flow[:,this_col]
            this_col_flows = [x for x in this_col_flows if x is not None]
            this_col_flows = np.array(this_col_flows)
            threshold_values = this_col_flows[this_col_flows >= magnitude_threshold_intial_observers]
            if(len(threshold_values) > 0):
                for temp in threshold_values:
                    this_col_threshold_values.append(temp)
        if(len(this_col_threshold_values)):
            print("No values with this threshold in col", this_col)
        all_cols_all_values.append(this_col_threshold_values)
    
    return all_cols_all_values

def transform_data_for_damping_plot(patch_flows,magnitude_threshold_intial_observers):
    nr_of_rows = len(patch_flows[0][:,0])
    nr_of_cols = len(patch_flows[0][0,:])
    all_rowwise_threshold_values = np.empty((nr_of_rows,nr_of_cols), dtype=np.ndarray)
    all_rowwise_threshold_values[:] = None
    for i in range(0,nr_of_rows):
        for j in range(0,nr_of_cols):
            if(patch_flows[0][i,j] is not None):
                all_rowwise_threshold_values[i,j] = np.array([])
    
    for this_row in range(0, nr_of_rows):
        for frame_count,patch_flow in enumerate(compressed_patch_flows):
            this_row_flows = patch_flow[this_row,:]
            this_row_flows = [x for x in this_row_flows if x is not None]
            for this_col, value in enumerate(this_row_flows):
                if (value >= magnitude_threshold_intial_observers):
                    all_rowwise_threshold_values[this_row,this_col] = np.append(all_rowwise_threshold_values[this_row,this_col],value)
    return all_rowwise_threshold_values

patch_flows, all_angs = read_data()
compressed_patch_flows = compress_data_at_patch_level(patch_flows)
all_cols_all_values = transform_data_for_box_plot(patch_flows,magnitude_threshold_intial_observers)
plot_box_plot_of_col_wise_mags(all_cols_all_values)
all_rowwise_threshold_values = transform_data_for_damping_plot(patch_flows,magnitude_threshold_intial_observers)

plot_damping_values_of_magnitude(all_rowwise_threshold_values,stat=0)
plot_damping_values_of_magnitude(all_rowwise_threshold_values,stat=1)
plot_damping_values_of_magnitude(all_rowwise_threshold_values,stat=2)


nr_of_rows = len(all_rowwise_threshold_values[:,0])
all_row_wise_stats = []
for this_row in range(0, nr_of_rows):
    threshold_values = all_rowwise_threshold_values[this_row,:]
    damping_values = []
    for value in threshold_values:
        if value is not None:
            try:
                damping_values.append((np.mean(value),np.median(value),np.max(value)))
            except ValueError:  #raised if `y` is empty.
                print("value error")
                pass

    all_row_wise_stats.append(damping_values)







# with open(join(os.getcwd(),output_all_peak_stats_folder,peak_stats_file), 'wb') as fp:
#     pickle.dump(all_row_wise_stats, fp)
#     print("Saved threshold values",peak_stats_file)


































#### old backup ####

# # -*- coding: utf-8 -*-
# """
# Created on Thu Jun 10 17:11:47 2021

# @author: Khamar Uz Zama
# """

# import pickle
# import A_Config
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import datetime

# from os.path import join
# from scipy.signal import find_peaks
# """
# This program:
#     1. Reads flows and angs
#     2. Detects peaks of angs and extracts the data in the window size
#     3. Creates the subplots of angs mags/flows
#     4. Creates box plot of all the mags
#     5. Extract medians from the box plots and saves them (Used in multi-liver plots)
#     6. Calculates the damping values of Magnitude's median
# """

# # BGR
# patch_colors_BGR = [(0, 0, 255), # Red
#                 (0, 255, 0), # Green
#                 (255, 0, 0),  # Blue
#                 (0, 200, 200),
#                 (200, 200, 0), 
#                 (200, 0, 200),
#                 (50, 100, 200),
#                 (100, 200, 50), 
#                 (200, 50, 100),
#                 (80, 50, 100)
#                 ]

# # Convert to RGB
# patch_colors_RGB = [(color[2],color[1],color[0]) for color in patch_colors_BGR]    
# patch_colors_RGB = np.array(patch_colors_RGB,float) / 255

# video_file = A_Config.video_file
# angs_file_initial_observers = A_Config.angs_file_initial_observers
# patch_flows_file = A_Config.patch_flows_file
# peak_stats_file = A_Config.peak_stats_file

# input_individual_patch_flows_folder = A_Config.individual_patch_flows_folder
# initial_observer_angs_folder = A_Config.initial_observer_angs_folder
# output_all_peak_stats_folder =  A_Config.all_peak_stats_folder
# output_all_peak_results_folder = A_Config.all_peak_results_folder
# this_video_results_folder = video_file[0:-4]

# if not os.path.isdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder)):
#     os.mkdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder))
# # else:
# #     filelist = [ f for f in os.listdir(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder))]
# #     for f in filelist:
# #         os.remove(join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder, f))

# def read_data():
#     """
#     Read Individual flows and angles of observers to sub plot the data
#     """
#     with open(join(os.getcwd(),input_individual_patch_flows_folder,patch_flows_file), "rb") as fp:
#         patch_flows = pickle.load(fp)
        
#     with open(join(os.getcwd(),initial_observer_angs_folder,angs_file_initial_observers), "rb") as fp:
#         angs = pickle.load(fp)
        
#     return patch_flows, angs

# def col_transformation(patch_flows):
#     all_cols_all_frames = []
#     # for col_count in range(len(patch_flows[0])):
#     for col_count in range(len(patch_flows[0][0])):        
#         this_col_all_frames = []
#         for frame_count in range(len(patch_flows)):
#             this_col_in_this_frame = patch_flows[frame_count][:,col_count]
#             this_col_in_this_frame = [x for x in this_col_in_this_frame if x is not None]
#             this_col_in_this_frame = np.average(this_col_in_this_frame)
#             this_col_all_frames.append(this_col_in_this_frame)
#         all_cols_all_frames.append(this_col_all_frames)
    
#     return all_cols_all_frames



# def create_single_row_all_obsvs_mag_ang_subplot(patch_flows,angs, smoothed_angs, all_angs,angle_peak_indices):
#     angs = [np.average(ang) for ang in angs]

#     fig = plt.figure()
#     gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    
#     # Subplot 1 - Flows
#     x = np.arange(len(patch_flows))
    

#     tot_rows = np.shape(peak_patch_flows[0])[0]
#     row_to_plot = 0
#     this_row_all_flows_all_frames = []
#     for frame_count,this_frame_patch_flow in enumerate(patch_flows):
#         this_row_in_this_frame = this_frame_patch_flow[row_to_plot]
#         this_row_in_this_frame = [np.average(x) for x in this_row_in_this_frame if x is not None]
#         this_row_all_flows_all_frames.append(this_row_in_this_frame)
    
#     zz = np.array(this_row_all_flows_all_frames).T.tolist()
    
#     ax0 = plt.subplot(gs[0])
#     lines = [None]*len(zz)
    
#     for i,patch_mags in enumerate(zz):
#         if(i%2==0):
#             lines[i], = ax0.plot(x, patch_mags, color=patch_colors_RGB[i], label=str(i))
#             for x_angle_peak in angle_peak_indices:
#                 ax0.scatter(x_angle_peak,patch_mags[x_angle_peak],marker='X',color='k')#patch_colors_RGB[i])
    
#     lines = [line for line in lines if line is not None]
#     lines = tuple(lines)
#     # ax0.legend(lines, ('Magnitudes'), loc='upper right')
#     ax0.legend(handles=lines, bbox_to_anchor=(1, 1), loc='upper left')

#     ax0.title.set_text('Magnitudes for all observers of row_id:'+str(row_to_plot)+' : '+video_file[:-4])
#     ax0.set_ylabel('Pixel Displacement')
    
#     # Subplot 2 - Angs
    
#     lines = [None]*len(zz)
#     ax1 = plt.subplot(gs[1], sharex = ax0)
    
#     for i in range(len(all_angs[0])):
#         lines[i], = ax1.plot(x, all_angs[:,i] , color=patch_colors_RGB[i])
    
#     smoothed_angs_line, = ax1.plot(x, smoothed_angs, color='b')#, linestyle='--')
#     # x = np.arange(len(angs))
#     # y = angs
#     # ax1 = plt.subplot(gs[1], sharex = ax0)
#     # angs_line, = ax1.plot(x, y, color='k')#, linestyle='--')
#     # smoothed_angs_line, = ax1.plot(x, smoothed_angs, color='b')#, linestyle='--')

#     for x_angle_peak in angle_peak_indices:
#         ax1.scatter(x_angle_peak,smoothed_angs[x_angle_peak],marker='X',c='k')
        
#     # ax1.legend((angs_line,smoothed_angs_line), ('Angle','Smoothed angle'), loc='upper left')
#     ax1.title.set_text('Angles')
#     ax1.set_xlabel('Frames')
#     ax1.set_ylabel('Angle of movement (Degrees)')

#     # remove vertical gap between subplots
#     plt.subplots_adjust(hspace=.0)
#     # plot_name = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())+"AngWMag.png"
#     plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"AngWSingleRowMag.png"

#     plot_save_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)    
#     # plot_save_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,"AngWMag.png")
#     plt.savefig(plot_save_path)
#     # plt.show()    
#     # plt.close()

#     fig.clf()
#     plt.clf()    
#     return


# def create_mag_ang_subplot(patch_flows,angs, smoothed_angs, all_angs,angle_peak_indices):
#     angs = [np.average(ang) for ang in angs]

#     fig = plt.figure()
#     gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    
#     # Subplot 1 - Flows
#     x = np.arange(len(patch_flows))
    

#     all_cols_all_frames = col_transformation(patch_flows)
    
#     ax0 = plt.subplot(gs[0])
#     lines = [None]*len(all_cols_all_frames[0])
    
#     for i,patch_mags in enumerate(all_cols_all_frames):
#         lines[i], = ax0.plot(x, patch_mags, color=patch_colors_RGB[i])
#         for x_angle_peak in angle_peak_indices:
#             ax0.scatter(x_angle_peak,patch_mags[x_angle_peak],marker='X',color='k')#patch_colors_RGB[i])
    
#     lines = tuple(lines)
#     #ax0.legend(lines, ('Magnitudes'), loc='upper right')
#     ax0.title.set_text('Magnitudes for '+video_file[:-4])
#     ax0.set_ylabel('Pixel Displacement')
    
#     # Subplot 2 - Angs
    
#     lines = [None]*len(all_cols_all_frames[0])
#     ax1 = plt.subplot(gs[1], sharex = ax0)
    
#     for i in range(len(all_angs[0])):
#         if(i==0):
#             lines[i], = ax1.plot(x, all_angs[:,i] , color=patch_colors_RGB[i])
#         else:
#             continue
    
#     smoothed_angs_line, = ax1.plot(x, smoothed_angs, color='b')#, linestyle='--')
#     # x = np.arange(len(angs))
#     # y = angs
#     # ax1 = plt.subplot(gs[1], sharex = ax0)
#     # angs_line, = ax1.plot(x, y, color='k')#, linestyle='--')
#     # smoothed_angs_line, = ax1.plot(x, smoothed_angs, color='b')#, linestyle='--')

#     for x_angle_peak in angle_peak_indices:
#         ax1.scatter(x_angle_peak,smoothed_angs[x_angle_peak],marker='X',c='k')
        
#     # ax1.legend((angs_line,smoothed_angs_line), ('Angle','Smoothed angle'), loc='upper left')
#     ax1.title.set_text('Angles')
#     ax1.set_xlabel('Frames')
#     ax1.set_ylabel('Angle of movement (Degrees)')

#     # remove vertical gap between subplots
#     plt.subplots_adjust(hspace=.0)
#     # plot_name = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())+"AngWMag.png"
#     plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"_AngWMag.png"

#     plot_save_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)    
#     # plot_save_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,"AngWMag.png")
#     plt.savefig(plot_save_path)
#     # plt.show()    
#     # plt.close()

#     fig.clf()
#     plt.clf()    
#     return

# def patch_artist(bp, ax_box_plot):

#     ax_box_plot.title.set_text('Distance vs Magnitude of '+video_file[:-4])
#     ax_box_plot.set_ylabel('Magnitude')
#     ax_box_plot.set_xlabel('Distance from liver boundary')

#     for box in bp['boxes']:
#         # change outline color
#         box.set( color='#7570b3', linewidth=2)
#         # change fill color
#         box.set( facecolor = '#1b9e77' )
    
#     ## change color and linewidth of the whiskers
#     for whisker in bp['whiskers']:
#         whisker.set(color='#7570b3', linewidth=2)
    
#     ## change color and linewidth of the caps
#     for cap in bp['caps']:
#         cap.set(color='#7570b3', linewidth=2)
    
#     ## change color and linewidth of the medians
#     for median in bp['medians']:
#         median.set(color='#b2df8a', linewidth=2)
    
#     ## change the style of fliers and their fill
#     for flier in bp['fliers']:
#         flier.set(marker='o', color='#e7298a', alpha=0.5)    


# def transform_data_to_column_wise_for_box_plot(peak_patch_flows):
#     """
#     Simply transforms the data from matrix to column wise for box plot only
#     """
#     all_cols_all_peak_frames = []
#     tot_cols = np.shape(peak_patch_flows[0])[1]
#     # for col_count in range(len(peak_patch_flows[0])):
#     for col_count in range(tot_cols):
#         this_col_all_peak_frames = []
#         for frame_count in range(len(peak_patch_flows)):
#             this_col_in_this_peak_frame = peak_patch_flows[frame_count][:,col_count]
#             this_col_in_this_peak_frame = [x for x in this_col_in_this_peak_frame if x is not None]
#             for each_patch_in_this_col in this_col_in_this_peak_frame:
#                 this_col_all_peak_frames.append(each_patch_in_this_col)
#         all_cols_all_peak_frames.append(this_col_all_peak_frames)
        
#     return all_cols_all_peak_frames

# def transform_data_to_row_wise_for_damping_plot(peak_patch_flows):
#     """
#     Simply transforms the data from matrix to row wise for damping plot only
#     """    
#     all_rows_all_peak_frames = []
#     tot_rows = np.shape(peak_patch_flows[0])[0]
    
#     for row_count in range(tot_rows):
#         this_row_all_peak_frames = []
#         for frame_count in range(len(peak_patch_flows)):
#             this_row_in_this_peak_frame = peak_patch_flows[frame_count][row_count]
#             this_row_in_this_peak_frame = [x for x in this_row_in_this_peak_frame if x is not None]
#             this_row_all_peak_frames.append(this_row_in_this_peak_frame)
#         all_rows_all_peak_frames.append(this_row_all_peak_frames)
    
#     return all_rows_all_peak_frames

# def smooth_angs(angs, harmonics=5):
#     rft = np.fft.rfft(angs)
#     rft[harmonics:] = 0   
#     smooth_angs = np.fft.irfft(rft)
    
#     return smooth_angs

# def plot_box_plot_of_col_wise_mags(all_col_wise_peak_values):
#     fig_box_plot = plt.figure(1, figsize=(9, 6))
#     ax_box_plot = fig_box_plot.add_subplot(111)
#     ax_box_plot.set_title("Box plot of " +video_file[:-4])
    
#     bp = ax_box_plot.boxplot(all_col_wise_peak_values, patch_artist = True)
#     patch_artist(bp,ax_box_plot)
    
#     plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"_BoxPlot.png"
#     box_plot_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)
#     plt.savefig(box_plot_path)
#     fig_box_plot.clf()
#     plt.clf()
    
#     return

# def plot_damping_values_of_magnitude(all_row_wise_peak_stats, stat=0):

    
#     fig = plt.figure()
#     ax0 = fig.add_subplot(111)

#     ax0.set_ylabel('% Decrease in magnitudes')
#     ax0.set_xlabel('Distance from Initial Observer')
    
#     if(stat == 0):
#         save_title = "Mean.png"
#         ax0.title.set_text('Damping of mean of magnitudes for '+video_file[:-4])
#     elif(stat == 1):
#         save_title = "Median.png"
#         ax0.title.set_text('Damping of median of magnitudes for '+video_file[:-4])
#     elif(stat == 2):
#         save_title = "Max.png"
#         ax0.title.set_text('Damping of max of magnitudes for '+video_file[:-4])

#     lines = [None]*len(all_row_wise_peak_stats)
    
#     for i,row_stats in enumerate(all_row_wise_peak_stats):
#         damping_values = [(temp[stat]/row_stats[0][stat])*100 for temp in row_stats]
#         x = np.arange(len(row_stats))
#         lines[i], = ax0.plot(x, damping_values, color=patch_colors_RGB[i])
        
#     plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p_")+save_title
#     plot_path = join(os.getcwd(),output_all_peak_results_folder,this_video_results_folder,plot_name)
#     plt.savefig(plot_path)
    
#     return

# def calculate_peak_stats(all_row_wise_peak_values):
#     all_row_wise_peak_stats = []
    
#     for row_wise_peak_values in all_row_wise_peak_values:
#         nr_of_patches_in_this_row = len(row_wise_peak_values[0])
#         this_row_stats = []
#         for patch_cnt in range(nr_of_patches_in_this_row):
#             all_peaks_of_this_patch = []
#             for peak in row_wise_peak_values:
#                 all_peaks_of_this_patch.append(peak[patch_cnt])
#             this_patch_peak_stats = ((np.mean(all_peaks_of_this_patch),np.median(all_peaks_of_this_patch),np.max(all_peaks_of_this_patch)))
#             this_row_stats.append(this_patch_peak_stats)
#         all_row_wise_peak_stats.append(this_row_stats)
    
#     return all_row_wise_peak_stats

# patch_flows, all_angs = read_data()
        
# # angs = [np.average(ang) for ang in angs]
# angs = np.asarray(all_angs)
# angs = angs[:,0] # Take only first obsv for now....
# # angs =     # Smooth mags for experimenting
# # all_cols_all_frames = col_transformation(patch_flows)
# # angs = all_cols_all_frames[0]

# smoothed_angs = smooth_angs(angs,harmonics=21)
# smoothed_angs = angs
# if(len(angs)-len(smoothed_angs)==1):
#     # Sometimes len(angs) is not same as len(smoothed_angs). Just append last value to make them equal and make it work x0
#     smoothed_angs = np.append(smoothed_angs, smoothed_angs[-1])

# # angle_peak_indices, _ = find_peaks(smoothed_angs, height=150.0, distance=1)
# angle_peak_indices, _ = find_peaks(smoothed_angs, height=(100.0,200), distance=10)

# # angle_peak_indices, _ = find_peaks(smoothed_angs, height=0.4, distance=25)
# peak_patch_flows = [patch_flows[i] for i in angle_peak_indices]
# zzzz = [smoothed_angs[i] for i in angle_peak_indices]

# # create_single_row_all_obsvs_mag_ang_subplot(patch_flows,angs, smoothed_angs, all_angs,angle_peak_indices)
 
# # Uncomment last line in this function to plot
# create_mag_ang_subplot(patch_flows, angs, smoothed_angs,all_angs, angle_peak_indices = angle_peak_indices)

# all_col_wise_peak_values = transform_data_to_column_wise_for_box_plot(peak_patch_flows)
# plot_box_plot_of_col_wise_mags(all_col_wise_peak_values)

# all_row_wise_peak_values = transform_data_to_row_wise_for_damping_plot(peak_patch_flows)
# all_row_wise_peak_stats = calculate_peak_stats(all_row_wise_peak_values)
# plot_damping_values_of_magnitude(all_row_wise_peak_stats,stat=0)
# plot_damping_values_of_magnitude(all_row_wise_peak_stats,stat=1)
# plot_damping_values_of_magnitude(all_row_wise_peak_stats,stat=2)


# # with open(join(os.getcwd(),output_all_peak_stats_folder,peak_stats_file), 'wb') as fp:
# #     pickle.dump(all_row_wise_peak_stats, fp)

