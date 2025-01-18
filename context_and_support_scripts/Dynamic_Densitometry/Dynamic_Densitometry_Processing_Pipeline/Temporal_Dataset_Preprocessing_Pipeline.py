# Robert Cooper
# 9-15-2017
# This script removes residual distortion from a strip-registered dataset.
# It requires:
#    * A *functioning* MATLAB runtime, that has been set up to link to Python (optional).
#    * The .dmp file output from Alfredo Dubra's Demotion software suite. **I realize this makes it VERY specific-
#      I do not promise any amazing things happening as result of using this software!**
#    * The 'mat' file corresponding to the grid calibration- also using Alf Dubra's script.
#    * The dataset you wish to put through the temporal analysis pipeline.
#
#

try:
    import matlab.engine # This needs to be imported first for some stupid reason.
except:
    import Tkinter as tk
    import Tkconstants, tkFileDialog, tkMessageBox
    import os, sys, ctypes
    import subprocess
    import socket
    
    options = {}
    options['title'] = 'Please select your [MATLABROOT]\extern\engines\python folder to link to MATLAB.'
    matlab_folder_path = tkFileDialog.askdirectory(**options)

    ctypes.windll.shell32.ShellExecuteW(None, u"runas",  unicode("C:\\Python27\\python.exe"), u"setup.py install", unicode(matlab_folder_path), 1)   

    try:
        import matlab.engine
    except:
        tkMessageBox.showerror("Linking (should be) successful!", "If the console did not display any errors, then linking successful! Please restart this script.")
        sys.exit(0)

import os, pickle
import Tkinter as tk
import Tkconstants, tkFileDialog, tkMessageBox
import numpy as np

root = tk.Tk()

try:
    mat_engi = matlab.engine.start_matlab()
except:
    tkMessageBox.showerror("Unable to start MATLAB! Ensure you have a valid copy of MATLAB installed AND it has been linked with python.")
    quit(1)


options = {}
options['title'] = 'Select the DESINUSOID FILE:'
options['parent'] = root
options['filetypes'] = [("MAT File", ".mat")]

desinsoid_file = tkFileDialog.askopenfilename(**options)

static_distortion = mat_engi.Static_Distortion_Repair(desinsoid_file)

just_the_dir = os.path.split(desinsoid_file)[0]

options = {}
options['title'] = 'Select the folder containing the DMP files:'
options['parent'] = root
options['initialdir'] = just_the_dir

dmp_folder_path = tkFileDialog.askdirectory(**options)

options = {}
options['title'] = 'Select the folder containing the IMAGE or MOVIE files:'
options['parent'] = root
options['initialdir'] = dmp_folder_path

image_folder_path = tkFileDialog.askdirectory(**options)

stimend = 38
stimbegin = 3

# progo = ttk.Progressbar(root, length=len(os.listdir(dmp_folder_path)))
# progo.pack()
fixed_images = []
THEpath = ""

for thisfile in os.listdir(dmp_folder_path):
    if thisfile.endswith(".dmp"):

        try:
            pickle_path = os.path.join(dmp_folder_path, thisfile)

            # Fix the fact that it was done originally in Windows...
            pickle_file = open(pickle_path, 'rb')
            text = pickle_file.read().replace('\r\n', '\n')
            pickle_file.close()

            pickle_file = open(pickle_path, 'wb')
            pickle_file.write(text)
            pickle_file.close()

            pickle_file = open(pickle_path, 'r')
            pick = pickle.load(pickle_file)

            ff_translation_info_rowshift = pick['full_frame_ncc']['row_shifts']
            ff_translation_info_colshift = pick['full_frame_ncc']['column_shifts']
            strip_translation_info = pick['sequence_interval_data_list']

            firsttime = True

            pickle_file.close()

            # Find the dmp's matching image(s).
            modalities = ('confocal', 'split_det', 'avg', 'visible')

            images_to_fix =[]
            # Find all images in our folder that this dmp applies to.
            for thismode in modalities:
                if thismode in thisfile:
                    #print "Found the modality of this dmp file! It is: "+thismode

                    for mode in modalities:
                        checkfile = thisfile[0:-4].replace(thismode, mode)

                        for imagefile in os.listdir(image_folder_path):
                            if (checkfile in imagefile) and (imagefile.endswith(".tif") or imagefile.endswith(".avi")):
                               # print("Whoa! " + imagefile + " matched!")
                                images_to_fix.append(imagefile)
                            #else:
                                #print("Whoa! " + imagefile + " didn't match " +checkfile)
                    break

            numgood = 0
				
            for index in pick['acceptable_frames']:
                if index >= stimbegin and index <= stimend:
                    numgood += 1
            
            #print("There are: "+str(len(images_to_fix))+" images to fix")
            print("There are: "+str(numgood)+" image indices within the stimulus duration...")
                    
            # If we don't have any accompanying images, just say fuck it and move on
            if images_to_fix and numgood >= (stimend-stimbegin)*0.6:
                print("Enough to continue the pipeline.")
                print("Using DMP file: " + thisfile)

                minmaxpix = np.zeros((len(strip_translation_info), 2))
                # print minmaxpix.shape

                i = 0
                for frame in strip_translation_info:
                    if len(frame) > 0:
                        ref_pixels = frame[0]['slow_axis_pixels_in_current_frame_interpolated']
                        minmaxpix[i, 0] = ref_pixels[0]
                        minmaxpix[i, 1] = ref_pixels[-1]
                        i += 1

                # print minmaxpix[:, 1].min()
                # print minmaxpix[:, 0].max()
                topmostrow = minmaxpix[:, 0].max()
                bottommostrow = minmaxpix[:, 1].min()

                # print np.array([pick['strip_cropping_ROI_2'][-1]])
                # The first row is the crop ROI.
                # np.savetxt(pickle_path[0:-4] + "_transforms.csv", np.array([pick['strip_cropping_ROI_2'][-1]]),
                #            delimiter=",", newline="\n", fmt="%f")

                shift_array = np.zeros([len(strip_translation_info)*3, 1000])
                shift_ind = 0
                for frame in strip_translation_info:
                    if len(frame) > 0:
                        # print "************************ Frame " + str(frame[0]['frame_index'] + 1) + "************************"
                        # print "Adjusting the rows...."
                        frame_ind = frame[0]['frame_index']

                        ff_row_shift = ff_translation_info_rowshift[frame_ind]
                        ff_col_shift = ff_translation_info_colshift[frame_ind]

                        # First set the relative shifts
                        row_shift = (np.subtract(frame[0]['slow_axis_pixels_in_reference_frame'],
                                                 frame[0]['slow_axis_pixels_in_current_frame_interpolated']))
                        col_shift = (frame[0]['fast_axis_pixels_in_reference_frame_interpolated'])

                        # These will contain all of the motion, not the relative motion between the aligned frames-
                        # So then subtract the full frame row shift
                        row_shift = np.add(row_shift, ff_row_shift)
                        col_shift = np.add(col_shift, ff_col_shift)

                        shift_array[shift_ind*3, 0:len(frame[0]['slow_axis_pixels_in_reference_frame'])] = frame[0]['slow_axis_pixels_in_reference_frame']
                        shift_array[shift_ind*3+1, 0:len(col_shift)] = col_shift
                        shift_array[shift_ind*3+2, 0:len(row_shift)] = row_shift

                        shift_ind += 1

                # progo.configure("Extracted the eye motion from the dmp file...")


                for image in images_to_fix:
                    if "confocal" in image:
                         print("Removing distortion from: "+image +"...")
                         anchorfile = mat_engi.Eye_Motion_Distortion_Repair_Pipl(image_folder_path, image, pick['strip_cropping_ROI_2'][-1],
                                                          shift_array.tolist(), static_distortion,nargout=3)                    
                         writtenfile = anchorfile[0:2]
                         cropregion = anchorfile[2]

                for image in images_to_fix:
                    if "confocal" not in image:
                         print("Removing distortion from: "+image +"...")
                         anchorfile = mat_engi.Eye_Motion_Distortion_Repair_Pipl(image_folder_path, image, pick['strip_cropping_ROI_2'][-1],
                                                          shift_array.tolist(), static_distortion, cropregion, nargout=3)                                             

                np.savetxt(os.path.join(writtenfile[1], thisfile[0:-4] + "_repaired_acceptable_frames.csv"),
                           pick['acceptable_frames'],
                           delimiter=',', fmt='%f')
                
                if "confocal" in writtenfile[0]:
                    print("Culling excess frames from: " + writtenfile[0] + "...")
                    try:
                        writtenfile = mat_engi.Densitometry_Automatic_Frame_Culler_Pipl(writtenfile[0], writtenfile[1], nargout=2)
                        fixed_images += [ writtenfile[0] ]
                        THEpath = writtenfile[1]
                    except(RuntimeError) as err:
                        print(err)
                        print("Failed to process video: " + writtenfile[0] + "!")
            else:
                print("Accompanying AVI not found, or not enough images in stimulus region to continue the pipeline.")

        except(ValueError, RuntimeError) as err:
            print(err)
            tkMessageBox.showwarning("DMP failed to process.",
                                     "Failed to process DMP (" + thisfile + ")! This file may be corrupted. Re-process the DMP, or contact your local RFC.")

# mat_engi.input_test(fixed_images, nargout=0)
print("Relativizing trials...")
mat_engi.Relativize_Trials_Pipl(fixed_images, THEpath, nargout=0)

root.destroy()
# shiftT = np.transpose(shift_array)
# transhandle = open(pickle_path[0:-4] + "_transforms.csv", 'w')
# np.savetxt(transhandle, shift_array, delimiter=',', fmt='%f')
# transhandle.close()
