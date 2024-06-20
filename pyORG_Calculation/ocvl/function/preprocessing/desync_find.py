from itertools import repeat
from pathlib import Path

import cv2
import numpy as np
import multiprocessing as mp
import os
from os import walk
from os.path import splitext
from tkinter import *
from tkinter import filedialog, simpledialog
from tkinter import ttk
from scipy.ndimage import binary_dilation
import pandas as pd
from matplotlib import pyplot as plt
from ocvl.function.preprocessing.improc import flat_field, weighted_z_projection, simple_image_stack_align, \
    optimizer_stack_align
from ocvl.function.utility.generic import GenericDataset, PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video


root = Tk()
root.lift()
w = 1
h = 1
x = root.winfo_screenwidth() / 4
y = root.winfo_screenheight() / 4
root.geometry(
    '%dx%d+%d+%d' % (
        w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)

if not pName:
    quit()

dataset = MEAODataset(file, analysis_modality=a_mode, ref_modality=ref_mode, stage=PipeStages.RAW)

dataset.load_raw_data()

num_vid_proj = ref_im_proj.shape[-1]
print("Selecting ideal central frame...")
dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(ref_im_proj.astype("uint8")),
                                                            repeat(np.ceil(weight_proj).astype("uint8")),
                                                            range(len(allFiles[loc]))))
shift_info = dist_res.get()

avg_loc_dist = np.zeros(len(shift_info))
f = 0
for allshifts in shift_info:
    # allshifts = simple_image_stack_align(vid.data, mask, f)
    allshifts = np.stack(allshifts)
    allshifts **= 2
    allshifts = np.sum(allshifts, axis=1)
    avg_loc_dist[f] = np.mean(np.sqrt(allshifts))  # Find the average distance to this reference.
    f += 1

avg_loc_idx = np.argsort(avg_loc_dist)
dist_ref_idx = avg_loc_idx[0]

