# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:03:18 2021

@author: Khamar Uz Zama
"""
import math

#### Folders ######
videos_folder = "0_Data"
flow_folder = "1_Flows" # Output of B_Create_Flows program
initial_observer_angs_folder = "2_Angs_Initial_Observers"
individual_patch_flows_folder = "3_Flows_Indiv_Observers" 
all_peak_stats_folder = "4_Magnitude_Peak_Stats"
all_peak_results_folder = "5_Peak_Results"
multi_liver_results_folder = "Multi-Liver"


#### Constants ########


#### Files ########

#### INKA002-RUN-11-F0/F1 ####
## Files ## 
# video_file = "INKA002-Run11.mp4"
# mask_video_file = "INKA002-Run11-Predicted.avi"
# flows_file = "INKA002-Run11-Flows.pickle"
# patch_flows_file = "INKA002-Run11_matrix_Patch_Flows.pickle"
# angs_file = "INKA002-Run11_Angs.pickle"
# peak_stats_file = "INKA002-Run11_F0_peak_stats.pickle"
# fibrosis_stage = "F0/F1"
# angs_file_initial_observers = "INKA002-Run11_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA002-Run11_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 3.0
# # ## Observers ##
# angle_of_movement = 207
# pixelsPerCM = 39 # 11
# buffer_to_left,buffer_to_right = 20,60
# buffer_to_up,buffer_to_down = 110,15
# distance_from_liver_bd_x = 10
# x,y = 300,320 # For E_Plot_AOM_Single_Observer - center of Obsv

#### INKA004-RUN-24-F2 ####
## Files ## 
# video_file = "INKA004-Run24.mp4"
# mask_video_file = "INKA004-Run24-Predicted.avi"
# flows_file = "INKA004-Run24-Flows.pickle"
# patch_flows_file = "INKA004-Run24_matrix_Patch_Flows.pickle"
# angs_file = "INKA004-Run24_Angs.pickle"
# peak_stats_file = "INKA004-Run24_F2_peak_stats.pickle"
# fibrosis_stage = "F2"
# angs_file_initial_observers = "INKA004-Run24_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA004-Run24_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 2.5
# ## Observers ##
# # angle_of_movement = 210
# pixelsPerCM = 36 # 12
# buffer_to_left,buffer_to_right = 20,70
# buffer_to_up,buffer_to_down = 70,10
# distance_from_liver_bd_x = 5
# x,y = 300,320 # For E_Plot_AOM_Single_Observer - center of Obsv


#### INKA005-RUN-14-F4 ####
## Files ## 
# video_file = "INKA005-Run14.mp4"
# mask_video_file = "INKA005-Run14-Predicted.avi"
# flows_file = "INKA005-Run14-Flows.pickle"
# patch_flows_file = "INKA005-Run14_matrix_Patch_Flows.pickle"
# angs_file = "INKA005-Run14_Angs.pickle"
# peak_stats_file = "INKA005-Run14_F4_peak_stats.pickle"
# fibrosis_stage = "F4"
# angs_file_initial_observers = "INKA005-Run14_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA005-Run14_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 3
# ## Observers ##
# # angle_of_movement = 210
# pixelsPerCM = 32 # 14
# buffer_to_left,buffer_to_right = 20,70
# buffer_to_up,buffer_to_down = 40,20
# distance_from_liver_bd_x = 2
# x,y = 400,320 # For E_Plot_AOM_Single_Observer - center of Obsv


### INKA007-RUN-31-F0/F1 ####
# Files ## 
# video_file = "INKA007-Run31.mp4"
# mask_video_file = "INKA007-Run31-Predicted.avi"
# flows_file = "INKA007-Run31-Flows.pickle"
# patch_flows_file = "INKA007-Run31_matrix_Patch_Flows.pickle"
# angs_file = "INKA007-Run31_Angs.pickle"

# angs_file_initial_observers = "INKA007-Run31_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA007-Run31_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 4
# peak_stats_file = video_file[:-4]+"_F0"+"_peak_stats.pickle"
# fibrosis_stage = "F0/F1"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 42 # 10
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 90,10
# distance_from_liver_bd_x = 15
# x,y=300,330

# #### INKA008-RUN-29-F2 ####
# ## Files ## 
# video_file = "INKA008-Run29.mp4"
# mask_video_file = "INKA008-Run29-Predicted.avi"
# flows_file = "INKA008-Run29-Flows.pickle"
# patch_flows_file = "INKA008-Run29_matrix_Patch_Flows.pickle"
# angs_file = "INKA008-Run29_Angs.pickle"
# angs_file_initial_observers = "INKA008-Run29_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA008-Run29_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 3
# peak_stats_file = video_file[:-4]+"_F2"+"_peak_stats.pickle"
# fibrosis_stage = "F2"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 42 # 10
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 60,20
# distance_from_liver_bd_x = 20
# x,y=300,330


#### INKA009-RUN-21-F4 ####
## Files ## 
# video_file = "INKA009-Run21.mp4"
# mask_video_file = "INKA009-Run21-Predicted.avi"
# flows_file = "INKA009-Run21-Flows.pickle"
# patch_flows_file = "INKA009-Run21_matrix_Patch_Flows.pickle"
# angs_file = "INKA009-Run21_Angs.pickle"
# angs_file_initial_observers = "INKA008-Run29_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA008-Run29_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 3
# peak_stats_file = video_file[:-4]+"_F4"+"_peak_stats.pickle"
# fibrosis_stage = "F4"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 39 # 11
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 60,40
# distance_from_liver_bd_x = 20
# x,y=350,300

#### INKA010-RUN-16-F0/F1 ####
## Files ## 
# video_file = "INKA010-Run16.mp4"
# mask_video_file = "INKA010-Run16-Predicted.avi"
# flows_file = "INKA010-Run16-Flows.pickle"
# patch_flows_file = "INKA010-Run16_matrix_Patch_Flows.pickle"
# angs_file_initial_observers = "INKA010-Run16_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA010-Run16_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 2.0

# peak_stats_file = video_file[:-4]+"_F0"+"_peak_stats.pickle"
# fibrosis_stage = "F0/F1"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 39 # 11
# buffer_to_left,buffer_to_right = 50,100
# buffer_to_up,buffer_to_down = 88,20
# distance_from_liver_bd_x = 15
# x,y=380,330

#### INKA011-RUN-14-F4 ####
## Files ## 
# video_file = "INKA011-Run14.mp4"
# mask_video_file = "INKA011-Run14-Predicted.avi"
# flows_file = "INKA011-Run14-Flows.pickle"
# patch_flows_file = "INKA011-Run14_matrix_Patch_Flows.pickle"
# angs_file_initial_observers = "INKA011-Run14_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA011-Run14_Initial Observer Mags.pickle"
# magnitude_threshold_intial_observers = 1.0

# peak_stats_file = video_file[:-4]+"_F4"+"_peak_stats.pickle"
# fibrosis_stage = "F0/F1"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 39 # 11
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 90,40
# distance_from_liver_bd_x = 25
# x,y=380,330


#### INKA012-RUN-13-F0/F1 ####
## Files ## 
# video_file = "INKA012-Run13.mp4"
# mask_video_file = "INKA012-Run13-Predicted.avi"
# flows_file = "INKA012-Run13-Flows.pickle"
# patch_flows_file = "INKA012-Run13_matrix_Patch_Flows.pickle"
# angs_file = "INKA012-Run13_Angs.pickle"
# angs_file_initial_observers = "INKA012-Run13_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA012-Run13_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 1.0
# peak_stats_file = video_file[:-4]+"_F0"+"_peak_stats.pickle"
# fibrosis_stage = "F0/F1"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 39 # 11
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 70,55
# distance_from_liver_bd_x = 10
# x,y=330,280


#### INKA013-RUN-12-F0/F1 ####
## Files ## 
# video_file = "INKA013-Run12.mp4"
# mask_video_file = "INKA013-Run12-Predicted.avi"
# flows_file = "INKA013-Run12-Flows.pickle"
# patch_flows_file = "INKA013-Run12_matrix_Patch_Flows.pickle"
# angs_file = "INKA013-Run12_Angs.pickle"
# angs_file_initial_observers = "INKA013-Run12_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA013-Run12_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 2.5
# peak_stats_file = video_file[:-4]+"_F0"+"_peak_stats.pickle"
# fibrosis_stage = "F0/F1"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 36 # 11
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 60,40
# distance_from_liver_bd_x = 10
# x,y=330,280



#### INKA014-RUN-17-F2 ####
## Files ## 
# video_file = "INKA014-Run17.mp4"
# mask_video_file = "INKA014-Run17-Predicted.avi"
# flows_file = "INKA014-Run17-Flows.pickle"
# patch_flows_file = "INKA014-Run17_matrix_Patch_Flows.pickle"
# angs_file = "INKA014-Run17_Angs.pickle"
# angs_file_initial_observers = "INKA014-Run17_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA014-Run17_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 0.0
# peak_stats_file = video_file[:-4]+"_F2"+"_peak_stats.pickle"
# fibrosis_stage = "F2"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 36 # 11
# buffer_to_left,buffer_to_right = 20,50
# buffer_to_up,buffer_to_down = 20,10
# distance_from_liver_bd_x = 1
# x,y=330,280



### INKA015-RUN-15-F3 ####
# Files ## 
# video_file = "INKA015-Run15.mp4"
# mask_video_file = "INKA015-Run15-Predicted.avi"
# flows_file = "INKA015-Run15-Flows.pickle"
# patch_flows_file = "INKA015-Run15_matrix_Patch_Flows.pickle"
# angs_file_initial_observers = "INKA015-Run15_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA015-Run15_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 0.0
# peak_stats_file = video_file[:-4]+"_F3"+"_peak_stats.pickle"
# fibrosis_stage = "F3"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 36 # 12
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 50,20
# distance_from_liver_bd_x = 25
# x,y = 300,320 # For E_Plot_AOM_Single_Observer - center of Obsv

### INKA015-RUN-14-F3 ####
# Files ## 
video_file = "INKA015-Run14.mp4"
mask_video_file = "INKA015-Run14-Predicted.avi"
flows_file = "INKA015-Run14-Flows.pickle"
patch_flows_file = "INKA015-Run14_matrix_Patch_Flows.pickle"
angs_file_initial_observers = "INKA015-Run14_Initial Observer Angs.pickle"
mags_file_initial_observers = "INKA015-Run14_Initial Observer Mags.pickle"

magnitude_threshold_intial_observers = 0.0
peak_stats_file = video_file[:-4]+"_F3"+"_peak_stats.pickle"
fibrosis_stage = "F3"
## Observers ##
angle_of_movement = 210
pixelsPerCM = 36 # 12
buffer_to_left,buffer_to_right = 20,120
buffer_to_up,buffer_to_down = 35,20
distance_from_liver_bd_x = 50
x,y = 300,320 # For E_Plot_AOM_Single_Observer - center of Obsv


### INKA016-RUN-19-F0/F1 ####
# Files ## 
# video_file = "INKA016-RUN19.mp4"
# mask_video_file = "INKA016-RUN19-Predicted.avi"
# flows_file = "INKA016-RUN-19-Flows.pickle"
# patch_flows_file = "INKA016-RUN-19_matrix_Patch_Flows.pickle"
# angs_file_initial_observers = "INKA016-RUN-19_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA016-RUN-19_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 1
# peak_stats_file = video_file[:-4]+"_F0"+"_peak_stats.pickle"
# fibrosis_stage = "F3"
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 36 # 12
# buffer_to_left,buffer_to_right = 20,120
# buffer_to_up,buffer_to_down = 70,40
# distance_from_liver_bd_x = 20
# x,y = 300,320 # For E_Plot_AOM_Single_Observer - center of Obsv


#### INKA017-RUN-13-F4 ####
## Files ## 
# video_file = "INKA017-Run13.mp4"
# mask_video_file = "INKA017-Run13-Predicted.avi"
# flows_file = "INKA017-Run13-Flows.pickle"
# patch_flows_file = "INKA017-Run13_matrix_Patch_Flows.pickle"
# angs_file = "INKA017-Run13_Angs.pickle"
# peak_stats_file = "INKA017-Run13_F4_peak_stats.pickle"
# fibrosis_stage = "F4"
# angs_file_initial_observers = "INKA017-Run13_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA017-Run13_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 1.5
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 32 # 14
# buffer_to_left,buffer_to_right = 20,80
# buffer_to_up,buffer_to_down = 0,20
# distance_from_liver_bd_x = 2


#### INKA018-RUN-36-F4 ####
## Files ## 
# video_file = "INKA018-RUN36.mp4"
# mask_video_file = "INKA018-RUN36-Predicted.avi"
# flows_file = "INKA018-RUN36-Flows.pickle"
# patch_flows_file = "INKA018-RUN36_matrix_Patch_Flows.pickle"
# angs_file = "INKA018-RUN36_Angs.pickle"
# peak_stats_file = "INKA018-RUN36_F0_peak_stats.pickle"
# fibrosis_stage = "F0"
# angs_file_initial_observers = "INKA018-RUN36_Initial Observer Angs.pickle"
# mags_file_initial_observers = "INKA018-RUN36_Initial Observer Mags.pickle"

# magnitude_threshold_intial_observers = 1.5
# ## Observers ##
# angle_of_movement = 210
# pixelsPerCM = 32 # 14
# buffer_to_left,buffer_to_right = 20,80
# buffer_to_up,buffer_to_down =35,15
# distance_from_liver_bd_x = 0









# ##################### Flags ###################
flg_vis_liv_boundary = False
flg_plt_magnitude = False
flg_save_flow_data_of_video = True
flg_save_flow_data_of_initial_observers = True
flg_save_flow_data_of_patches = True

#### Vars ######
# add vars for creating patches here
