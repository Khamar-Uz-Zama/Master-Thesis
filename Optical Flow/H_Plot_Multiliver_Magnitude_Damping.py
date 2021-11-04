# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:08:52 2021

@author: Khamar Uz Zama
This program is used for creating plots of damping 
"""
import pickle
import A_Config
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from os.path import join


input_all_peak_stats_folder =  A_Config.all_peak_stats_folder
video_fibrosis_stage = A_Config.fibrosis_stage
output_all_peak_results_folder = A_Config.all_peak_results_folder
output_multi_liver_results_folder = A_Config.multi_liver_results_folder

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

def read_data():
    """
    Read Individual median values as a dictionary
    """
    
    from os import listdir
    from os.path import isfile
    mypath = join(os.getcwd(),input_all_peak_stats_folder)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    peak_stats = {}
    
    for stats_file in onlyfiles:
        print(stats_file)
        with open(join(mypath,stats_file), "rb") as fp:
             peak_stats[str(stats_file)] = pickle.load(fp)
    
    return peak_stats


def create_plot_lines_for_damping_values_of_magnitude(ax,video_file, all_row_wise_peak_stats, color, stat):
        
    lines = [None]*len(all_row_wise_peak_stats)
    for i,row_stats in enumerate(all_row_wise_peak_stats):
        damping_values = [(temp[stat]/row_stats[0][stat])*100 for temp in row_stats]
        x = np.arange(len(row_stats))
        lines[i], = ax.plot(x, damping_values, color=color)
        
        
    
    return lines


def plot_multi_liver_damping():
    # stats = [0,1,2]
    stats = [0]
    
    stat_text = ""
    fibrosis_stages = ["F0/F1","F2","F3","F4"]
    for stat in stats: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.set_ylabel('% Decrease in magnitudes')
        ax.set_xlabel('Distance from initial observer')
        
        if(stat == 0):
            ax.title.set_text('Multi-liver damping of mean of magnitudes')
            stat_text = "Mean"
        elif(stat == 1):
            ax.title.set_text('Multi-liver damping of median of magnitudes')
            stat_text = "Median"        
        elif(stat == 2):
            ax.title.set_text('Multi-liver damping of max of magnitudes')
            stat_text = "Max"    
        video_lines = []
        video_files = []
        i=0
        
        for file, peak_stat in peak_stats.items():
            video_fibrosis_stage = file[14:16]
            if(int(video_fibrosis_stage[1]) != 3):
                continue
            video_file = file[:16]
            video_file = "Pid_"+video_file[4:-8]+video_fibrosis_stage
            video_files.append(video_file)
            
            # if(int(video_fibrosis_stage[1]) != 3):
            #     color = patch_colors_RGB[1]
            
    
            color = patch_colors_RGB[i]
            # color = patch_colors_RGB[int(video_fibrosis_stage[1])]
            
                
            lines = create_plot_lines_for_damping_values_of_magnitude(ax,video_file=video_file, all_row_wise_peak_stats=peak_stat, color=color, stat=stat,)
            video_lines.append(lines[0])
        
            i+=1
            
        # ax.legend(video_lines, ['Run13_F4','Run16_F0'], loc='upper right')
        ax.legend(video_lines, video_files, loc='upper right')
        
        plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"Multiliver Damping plot"+stat_text
        plot_path = join(os.getcwd(),output_all_peak_results_folder,output_multi_liver_results_folder,plot_name)
        plt.savefig(plot_path)
        

def plot_all_livers():    
    stages_and_stats = {"0":[],
                        "2":[],
                        "3":[],
                        "4":[]
                        }
    
    # Group livers by fibrotic stages
    for file, peak_stat in peak_stats.items():
        video_fibrosis_stage = file[14:16]
        video_fibrosis_stage = video_fibrosis_stage[1]
        stages_and_stats[video_fibrosis_stage].append(peak_stat)                     
    
    
    lines = [None]*4
    stat = 0 # indicates mean of peak stat
    count = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stage_lines = []
    for stage, this_stage_all_livers in stages_and_stats.items():
        this_stage_all_damping = []
        for this_stage_this_liver in this_stage_all_livers:
            for i,trajectories in enumerate(this_stage_this_liver):
                for trajectory in trajectories:
                    damping_values = [(temp[stat]/trajectories[0][stat])*100 for temp in trajectories]
                this_stage_all_damping.append(damping_values)
        
        # Convert list of lists into 2d-numpy array of equal length filled with NoneType
        length = max(map(len, this_stage_all_damping))
        this_stage_all_damping = np.array([xi+[None]*(length-len(xi)) for xi in this_stage_all_damping])
        line = []
        # Remove all Nones and take average
        for i in range(0,len(this_stage_all_damping[0])):
            line.append(np.average([i for i in this_stage_all_damping[:,i] if i != None]))
        stage_lines.append(line)
        color = patch_colors_RGB[int(stage)]
        x = np.arange(len(this_stage_all_damping[0]))
        
        print(stage)
        lines[count], = ax.plot(x, line, color=color)
        count += 1
    ax.legend(lines, ["F0/F1","F2","F3","F4"], loc='upper right')
    ax.set_ylabel('% Decrease in magnitudes')
    ax.set_xlabel('Distance from initial observer')
    plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"All stages mean Damping plot"
    plot_path = join(os.getcwd(),output_all_peak_results_folder,output_multi_liver_results_folder,plot_name)
    plt.savefig(plot_path)
    
    
    for x in stage_lines:
        print(np.average(x))
    
    return  



def patch_artist(bp, ax_box_plot):



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




def plot_box_plot_of_damping_values(peak_stats):
                    
    stages_and_stats = {"0":[],
                        "2":[],
                        "3":[],
                        "4":[]
                        }
    
    
    # Group livers by fibrotic stages
    for file, peak_stat in peak_stats.items():
        video_fibrosis_stage = file[14:16]
        video_fibrosis_stage = video_fibrosis_stage[1]
        stages_and_stats[video_fibrosis_stage].append(peak_stat)                     
    stat = 0 # indicates mean of peak stat
    all_stages_all_damping = {"0":[],
                        "2":[],
                        "3":[],
                        "4":[]
                        }
    for stage, this_stage_all_livers in stages_and_stats.items():
        for this_stage_this_liver in this_stage_all_livers:
            for i,trajectories in enumerate(this_stage_this_liver):
                for trajectory in trajectories:
                    for temp in trajectories:
                        all_stages_all_damping[stage].append((temp[stat]/trajectories[0][stat])*100)
    

    all_col_wise_peak_values = list(all_stages_all_damping.values())
    all_col_wise_peak_values = np.array(all_col_wise_peak_values, dtype=object)
    
    fig_box_plot = plt.figure(1, figsize=(9, 6))
    ax_box_plot = fig_box_plot.add_subplot(111)
    
    bp = ax_box_plot.boxplot(all_col_wise_peak_values,positions=np.arange(0,4).tolist(),patch_artist = True) 
    patch_artist(bp,ax_box_plot)
    plt.xticks([0,1, 2, 3], ["F0/F1","F2","F3","F4"])


    ax_box_plot.set_ylabel('% Decrease in magnitudes')
    ax_box_plot.set_xlabel('Fibrosis Stage')    


    plot_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"All stages Damping box plot"
    plot_path = join(os.getcwd(),output_all_peak_results_folder,output_multi_liver_results_folder,plot_name)
    plt.savefig(plot_path)
    fig_box_plot.clf()
    plt.clf()    
    
    return all_stages_all_damping
    
    
peak_stats = read_data()
plot_multi_liver_damping()


# plot_all_livers()
# all_stages_all_damping = plot_box_plot_of_damping_values(peak_stats)

# np.average(all_stages_all_damping["0"])