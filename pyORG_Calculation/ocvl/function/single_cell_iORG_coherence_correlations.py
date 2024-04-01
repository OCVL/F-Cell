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

plt.figure(2)
plt.plot(cell_pwr_iORG_1.iloc[max_loc])
plt.plot(cell_pwr_iORG_2.iloc[max_loc])
plt.legend(labels = ["760nm", "Conf"])
plt.title('Full Length Max Correlation')

plt.figure(3)
plt.plot(cell_pwr_iORG_1.iloc[med_loc])
plt.plot(cell_pwr_iORG_2.iloc[med_loc])
plt.legend(labels = ["760nm", "Conf"])
plt.title('Full Length Median Correlation')

plt.figure(4)
plt.hist(testCorrW)
plt.xlabel('Pearsons correlation')
plt.ylabel('Count')
plt.title('Full Length Correlation Histogram')

plt.figure(5)
plt.hist(testCorrW_prestim)
plt.xlabel('Pearsons correlation')
plt.ylabel('Count')
plt.title('Prestim Correlation Histogram')

plt.figure(6)
plt.hist(testCorrW_poststim)
plt.xlabel('Pearsons correlation')
plt.ylabel('Count')
plt.title('Poststim Correlation Histogram')

plt.show()

print(' ')
cell_pwr_iORG_2 = pd.read_csv(fName_2)
print('fiheoiusfhoighoge')
