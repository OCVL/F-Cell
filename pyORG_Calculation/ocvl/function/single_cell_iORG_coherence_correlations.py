# Creating a file to calculate correlation coefficient between individual cell signals acquired under different
# imaging conditions


# Importing all the things...
import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
import matplotlib
import re
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles
from datetime import datetime, date, time, timezone

# User selects dataset from file
root = Tk()
fName_1 = filedialog.askopenfilename(title="Select the Stimulus_cell_power_iORG csv for the first condition.", parent=root)
print('selected path: ' + fName_1)

if not fName_1:
    quit()


fName_2 = filedialog.askopenfilename(title="Select the Stimulus_cell_power_iORG csv for the second condition.", parent=root)
print('selected path: ' + fName_2)

if not fName_2:
    quit()

stimName = filedialog.askopenfilename(title="Select the stimulus train file.", parent=root)
print('selected path: ' + stimName)

if not stimName:
    quit()

# TODO: have user select their output directory instead of doing this hardcoded crap...
# Splitting the fName_1 string in order to create a results directory for saving figure and csv outputs
fName_1_split = fName_1.split('/')
fName_1_sliced = fName_1_split[0:10:1] #This probs shouldn't be hardcoded but alas I'm lazy
fName_1_joined = '/'.join(fName_1_sliced)
out_dir = Path(fName_1_joined)
trial_type = fName_1_split[11]

dt = datetime.now()
now_timestamp = dt.strftime("%Y_%m_%d_%H_%M_%S")

# Reading in datasets as dataframe
cell_pwr_iORG_1 = pd.read_csv(fName_1)
cell_pwr_iORG_2 = pd.read_csv(fName_2)
stimTrain = pd.read_csv(stimName, header=None)

# Creating a truncated dataframe that only includes frames during and after stimulus
prestim_ind = stimTrain.iloc[0,0] - 21 # Chose 21 to match the number of pre and post stim frames from indvidual cell iORG script
poststim_ind = stimTrain.iloc[0,0] + 21

cell_pwr_iORG_1_prestim = cell_pwr_iORG_1.iloc[:, prestim_ind:stimTrain.iloc[0,0]]
cell_pwr_iORG_2_prestim = cell_pwr_iORG_2.iloc[:, prestim_ind:stimTrain.iloc[0,0]]
cell_pwr_iORG_1_poststim = cell_pwr_iORG_1.iloc[:, stimTrain.iloc[0,0]:poststim_ind]
cell_pwr_iORG_2_poststim = cell_pwr_iORG_2.iloc[:, stimTrain.iloc[0,0]:poststim_ind]

# calculating correlation coefficient
testCorrW = cell_pwr_iORG_1.corrwith(cell_pwr_iORG_2, axis=1, drop=False, method='pearson')
testCorrW_prestim = cell_pwr_iORG_1_prestim.corrwith(cell_pwr_iORG_2_prestim, axis=1, drop=False, method='pearson')
testCorrW_poststim = cell_pwr_iORG_1_poststim.corrwith(cell_pwr_iORG_2_poststim, axis=1, drop=False, method='pearson')

# plotting the signals from the highest and lowest correlation coeffs
print('Min Pearson correlation: %.5f' % testCorrW.min())
print('Median Pearson correlation: %.5f' % testCorrW.median())
print('Max Pearson correlation: %.5f' % testCorrW.max())
print(' ')
print('Mean Full Length Pearson correlation: %.5f' % testCorrW.mean())
print('Mean Pre Stim Pearson correlation: %.5f' % testCorrW_prestim.mean())
print('Mean Post Stim Pearson correlation: %.5f' % testCorrW_poststim.mean())

# saving stuff to a dataframe so it can be output to csv
corr_results_summary = pd.DataFrame({"Pearsons correlation": ['min full','median full', 'max full', 'mean full', 'mean prestim', 'mean poststim'],
                                     "Value": [testCorrW.min(), testCorrW.median(), testCorrW.max(), testCorrW.mean(), testCorrW_prestim.mean(), testCorrW_poststim.mean()]})


# calculating median manually because something is weird with finding that index
sort_testCorrW = testCorrW.sort_values(axis=0)
length_testCorrW = testCorrW.size
calc_med = round(length_testCorrW/2)
med_loc = testCorrW.index.get_loc(sort_testCorrW[sort_testCorrW == sort_testCorrW.iloc[calc_med]].index[0])

min_loc = int(testCorrW.idxmin())
max_loc = int(testCorrW.idxmax())

plt.figure(1)
plt.plot(cell_pwr_iORG_1.iloc[min_loc])
plt.plot(cell_pwr_iORG_2.iloc[min_loc])
plt.legend(labels = ["760nm", "Conf"])
plt.title('Full Length Min Correlation')
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_min_corr_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_min_corr_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())


fig, ax1 = plt.subplots()
ax1.plot(cell_pwr_iORG_1.iloc[min_loc], label='760nm', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(cell_pwr_iORG_2.iloc[min_loc], label='Conf', color='tab:orange')
plt.title('Full Length Min Correlation')
fig.legend(loc='upper right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_min_corr_adjustedAxes_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_min_corr_adjustedAxes_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

plt.figure(2)
plt.plot(cell_pwr_iORG_1.iloc[max_loc])
plt.plot(cell_pwr_iORG_2.iloc[max_loc])
plt.legend(labels = ["760nm", "Conf"])
plt.title('Full Length Max Correlation')
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_max_corr_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_max_corr_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

fig, ax1 = plt.subplots()
ax1.plot(cell_pwr_iORG_1.iloc[max_loc], label='760nm', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(cell_pwr_iORG_2.iloc[max_loc], label='Conf', color='tab:orange')
plt.title('Full Length Max Correlation')
fig.legend(loc='upper right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_max_corr_adjustedAxes_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_max_corr_adjustedAxes_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

plt.figure(3)
plt.plot(cell_pwr_iORG_1.iloc[med_loc])
plt.plot(cell_pwr_iORG_2.iloc[med_loc])
plt.legend(labels = ["760nm", "Conf"])
plt.title('Full Length Median Correlation')
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_median_corr_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_median_corr_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

fig, ax1 = plt.subplots()
ax1.plot(cell_pwr_iORG_1.iloc[med_loc], label='760nm', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(cell_pwr_iORG_2.iloc[med_loc], label='Conf', color='tab:orange')
plt.title('Full Length Median Correlation')
fig.legend(loc='upper right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_med_corr_adjustedAxes_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_med_corr_adjustedAxes_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

plt.figure(4)
plt.hist(testCorrW)
plt.xlabel('Pearsons correlation')
plt.ylabel('Count')
plt.title('Full Length Correlation Histogram')
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_stim_hist_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_full_length_stim_hist_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

plt.figure(5)
plt.hist([testCorrW_prestim,testCorrW_poststim], label=['Pre-stimulus','Post-stimulus'], histtype = 'step')
plt.xlabel('Pearsons correlation')
plt.ylabel('Count')
plt.title('Correlation Histogram')
plt.legend(loc='upper right')
plt.show(block=False)
plt.savefig(out_dir.joinpath(trial_type + "_pre_v_post_stim_hist_" + now_timestamp + ".svg"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.savefig(out_dir.joinpath(trial_type + "_pre_v_post_stim_hist_" + now_timestamp + ".png"),
            transparent=False, dpi=72, bbox_inches = "tight", pad_inches = 0)
plt.close(plt.gcf())

# Outputting csv results
summary_csv_dir = out_dir.joinpath(trial_type + "_Correlations_Summary_" + now_timestamp + ".csv")
corr_results_summary.to_csv(summary_csv_dir, index=False)

testCorrW_csv_dir = out_dir.joinpath(trial_type + "_FullLength_Correlation_Raw_" + now_timestamp + ".csv")
testCorrW.to_csv(testCorrW_csv_dir, index=False, header=False)

testCorrW_prestim_csv_dir = out_dir.joinpath(trial_type + "_PreStim_Correlation_Raw_" + now_timestamp + ".csv")
testCorrW_prestim.to_csv(testCorrW_prestim_csv_dir, index=False, header=False)

testCorrW_poststim_csv_dir = out_dir.joinpath(trial_type + "_PostStim_Correlation_Raw_" + now_timestamp + ".csv")
testCorrW_poststim.to_csv(testCorrW_poststim_csv_dir, index=False, header=False)

print(' ')
print('Done!')
