from pathlib import Path
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ocvl.function.preprocessing.improc import optimizer_stack_align
from src.ocvl.function.utility.resources import save_tiff_stack

if __name__ == "__main__":

    root = Tk()
    root.lift()
    w = 256
    h = 128
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.


    ref_fName = filedialog.askopenfilename(title="Select the reference image.", parent=root)

    if not ref_fName:
        quit()

    targ_fName = filedialog.askopenfilename(title="Select the target image.", parent=root)

    if not targ_fName:
        quit()

    targ_coords_fName = filedialog.askopenfilename(title="Select the target coordinates.", parent=root)

    if not targ_coords_fName:
        quit()


    ref_im = cv2.imread(ref_fName, cv2.IMREAD_GRAYSCALE)
    tar_im = cv2.imread(targ_fName, cv2.IMREAD_GRAYSCALE)
    coordlist = pd.read_csv(targ_coords_fName, header=None, encoding="utf-8-sig").to_numpy()

    # @TODO THIS IS TEMPORARY
    ref_im = ref_im[16:, :]

    im_stk = np.dstack((ref_im, tar_im))

    #aligned_im, xforms, inliers = relativize_image_stack(im_stk, dropthresh=0)
    aligned_im, xforms, inliers, _ = optimizer_stack_align(im_stk, im_stk>0, 0)

    aug_xform = np.linalg.pinv(np.vstack( (xforms[1], np.array([[0,0,1]]) ) ))
    aug_coords = np.hstack((coordlist, np.ones((coordlist.shape[0], 1))) ).T
    xformed_coords = np.dot(aug_xform, aug_coords).T

    #xformed_coords -= 1

    xformed_coords = xformed_coords[:, 0:2]

    goodcoords = xformed_coords[:, 0] >= 0
    goodcoords = (xformed_coords[:, 1] >= 0) & goodcoords
    goodcoords = (xformed_coords[:, 0] < ref_im.shape[1]) & goodcoords
    goodcoords = (xformed_coords[:, 1] < ref_im.shape[0]) & goodcoords

    plt.figure()
    plt.imshow(aligned_im[..., 0])
    plt.plot(xformed_coords[goodcoords, 0], xformed_coords[goodcoords, 1], 'b*')
    plt.plot(xformed_coords[~goodcoords, 0], xformed_coords[~goodcoords, 1], 'r*')
    plt.show(block=False)
    plt.waitforbuttonpress()

    ref_fName = Path(ref_fName)
    targ_coords_fName = Path(targ_coords_fName)

    xformed_coords[:,1] += 16

    np.savetxt(targ_coords_fName.with_stem(targ_coords_fName.stem + "_trimmed"), coordlist[goodcoords,:], fmt="%.3f", delimiter=",")
    np.savetxt(ref_fName.with_name( ref_fName.stem+"_coords").with_suffix(".csv"), xformed_coords[goodcoords, :], fmt="%.3f", delimiter=",")
    save_tiff_stack(ref_fName.with_stem( ref_fName.stem+"_allaligned"), aligned_im)