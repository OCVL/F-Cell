from pathlib import Path

import scipy
from matplotlib import pyplot, pyplot as plt
import cv2
import numpy as np

from tkinter import *
from tkinter import filedialog, ttk


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

    horz_conf_fName = filedialog.askopenfilename(title="Select the horizontally oriented image.", parent=root)
    if not horz_conf_fName:
        quit()

    vert_conf_fName = filedialog.askopenfilename(title="Select the vertical oriented image.", parent=root)
    if not vert_conf_fName:
        quit()

    horzconf_im = cv2.imread(horz_conf_fName, cv2.IMREAD_GRAYSCALE)
    vertconf_im = cv2.imread(vert_conf_fName, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create(5000, nOctaveLayers=55, enable_precise_upscale=True)

    horz_kp, horz_des = sift.detectAndCompute(horzconf_im, None)
    vert_kp, vert_des = sift.detectAndCompute(vertconf_im, None)

    matcher = cv2.BFMatcher_create(normType = cv2.NORM_L2)
    matches = matcher.knnMatch(horz_des, vert_des, k=2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    xform=[]
    if len(good_matches) >= 4:
        src_pts = np.float32([horz_kp[f1.queryIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([vert_kp[f1.trainIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)

        xform, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts,ransacReprojThreshold=2,  confidence=0.99, refineIters=1000)

        print(xform)
        if xform is not None and np.sum(inliers) >= 4:
            xformed_vert = cv2.warpAffine(vertconf_im, xform, vertconf_im.shape,
                                          flags=cv2.INTER_LINEAR)

            # -- Draw matches
            img_matches = np.empty((max(horzconf_im.shape[0], vertconf_im.shape[0]),
                                    horzconf_im.shape[1] + vertconf_im.shape[1], 3), dtype=np.uint8)
            cv2.drawMatches(horzconf_im, horz_kp, vertconf_im, vert_kp, good_matches, img_matches,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # -- Show detected matches
            cv2.imshow('Good Matches', img_matches)
            cv2.waitKey()
#            cv2.imwrite("output.tif",xformed_vert)
        else:
            print("Failed to find a good match, exiting...")
            quit()

    # Now process our split data.
    p = np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659], ndmin=2)
    d = np.array([0.109604, 0.276691, 0.00000, -0.276691, -0.109604], ndmin=2)
    horz_kern = p.T @ d
    horz_flip_kern = np.fliplr(horz_kern)  # This is flipped because our subtraction for vertical split is flipped.
    vert_kern = (p.T @ d).T
    vert_flip_kern = np.flipud(vert_kern)  # This is flipped because our subtraction for vertical split is flipped.

    horz_split_fName= horz_conf_fName.replace("Confocal", "CalculatedSplit")
    vert_split_fName = vert_conf_fName.replace("Confocal", "CalculatedSplit")

    horz_im = cv2.imread(horz_split_fName, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    vert_im = cv2.imread(vert_split_fName, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    xformed_vert = cv2.warpAffine(vert_im, xform, vert_im.shape,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)

    nancols = np.any(np.isfinite(xformed_vert), axis=0)
    leftind = 0
    rightind = len(nancols)

    if np.any(nancols):
        leftind = np.argmax(nancols)
        rightind = len(nancols) - np.argmax(nancols[::-1])

        # if leftind == rightind and np.all(~nancols[leftind:]):
        #     rightind = len(nancols)
        # elif leftind == rightind and np.all(~nancols[:rightind]):
        #     leftind = 0

    nanrows = np.any(np.isfinite(xformed_vert), axis=1)
    topind = 0
    bottomind = len(nanrows)

    if np.any(nanrows):
        topind = np.argmax(nanrows)
        bottomind = len(nanrows) - np.argmax(nanrows[::-1])

        # if topind == bottomind and np.all(~nanrows[topind:]):
        #     bottomind = len(nanrows)
        # elif topind == bottomind and np.all(~nanrows[:bottomind]):
        #     topind = 0

    # Crop to the only good area.
    #horz_im = horz_im[topind:bottomind, leftind:rightind]
    #xformed_vert = xformed_vert[topind:bottomind, leftind:rightind]

    dIx_dx = scipy.signal.convolve2d(horz_im, horz_kern, boundary='symmetric', mode='same')
    dIx_dy = scipy.signal.convolve2d(horz_im, vert_kern, boundary='symmetric', mode='same')
    dIy_dx = scipy.signal.convolve2d(xformed_vert, horz_flip_kern, boundary='symmetric', mode='same')
    dIy_dy = scipy.signal.convolve2d(xformed_vert, vert_flip_kern, boundary='symmetric', mode='same')

    cutoff = 4
    dIx_dx[np.abs(dIx_dx) > cutoff] = cutoff
    dIx_dy[np.abs(dIx_dy) > cutoff] = cutoff
    dIy_dx[np.abs(dIy_dx) > cutoff] = cutoff
    dIy_dy[np.abs(dIy_dy) > cutoff] = cutoff

    ax_of_stig = (dIx_dy + dIy_dx) / (dIx_dx - dIy_dy)
    angle_of_stig = np.atan(ax_of_stig)/2
    mean_sphere_lens_power = (dIx_dx + dIy_dy)/2

    x_orientmap = np.atan2(dIx_dy, dIx_dx)
    y_orientmap = np.atan2(dIy_dy, dIy_dx)
    combo_orientmap = np.atan2((dIx_dy + dIy_dx) , (dIx_dx - dIy_dy))

    mean_sphere_lens_power[np.isnan(mean_sphere_lens_power)] = 0
    combo_orientmap[np.isnan(combo_orientmap)] = 0

    leftside = np.vstack((np.hstack((np.ones_like(combo_orientmap), np.sin(combo_orientmap)**2)),
                          np.hstack((np.ones_like(combo_orientmap), np.cos(combo_orientmap)**2)),
                          np.hstack((np.zeros_like(combo_orientmap), (-np.sin(combo_orientmap)*np.cos(combo_orientmap)) )) ))
    rightside = np.vstack((dIx_dx, dIy_dy, (dIx_dy + dIy_dx)/2))
    rightside[np.isnan(rightside)] = 0
    SC = np.linalg.pinv(leftside) @ rightside
    SC_lsq = np.linalg.lstsq(leftside, rightside, rcond=None)[0]

    plt.figure("Dominant Orientation")
    plt.subplot(1, 3, 1)
    plt.hist(x_orientmap.flatten() * (180 / np.pi), bins=180)
    plt.subplot(1, 3, 2)
    plt.hist(y_orientmap.flatten() * (180 / np.pi), bins=180)
    plt.subplot(1, 3, 3)
    plt.hist(combo_orientmap.flatten() * (180 / np.pi), bins=180)


    plt.figure("Derivatives")
    plt.subplot(2,2,1)
    plt.imshow(dIx_dx, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(dIx_dy, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(dIy_dx, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(dIy_dy, cmap='gray')


    plt.figure("Mean sphere and angle of astigmatism")
    plt.subplot(1,2,1)
    plt.imshow(mean_sphere_lens_power, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(combo_orientmap, cmap='hsv')


    plt.figure("Most negative and most positive meridia")
    plt.subplot(1, 2, 1)
    plt.imshow(SC_lsq[0:720,:], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(SC_lsq[720:,:], cmap='gray')
    plt.show(block=False)
    plt.waitforbuttonpress()

    horz_split_path = Path(horz_split_fName)
    horz_split_path = horz_split_path.parent.joinpath("Results", horz_split_path.stem + "_meansphere.tif")
    horz_split_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(horz_split_path), mean_sphere_lens_power)


    print("PK FIRE")
    # pName = filedialog.askdirectory(title="Select the folder containing all videos of interest.", parent=root)
    # #pName = "P:\\RFC_Projects\\F-Cell_Generalization_Test_Data"
    # if not pName:
    #     quit()
    #
    #
    # root.update()
    #
    # # We should be 3 levels up from here. Kinda jank, will need to change eventually
    # config_path = Path(os.path.dirname(__file__)).parent.parent.parent.joinpath("config_files")
    #
    # json_fName = filedialog.askopenfilename(title="Select the configuration json file.", initialdir=config_path, parent=root)
    # if not json_fName:
    #     quit()

    # with mp.Pool(processes=int(np.round(mp.cpu_count()/2 ))) as pool:
    #
    #     dat_form, allData = parse_file_metadata(json_fName, pName, "processed")
    #
    #     processed_dat_format = dat_form.get("processed")
    #     pipeline_params = processed_dat_format.get("pipeline_params")
    #     modes_of_interest = pipeline_params.get(PipelineParams.MODALITIES)
    #
    #     output_folder = pipeline_params.get(PipelineParams.OUTPUT_FOLDER)
    #     if output_folder is None:
    #         output_folder = PurePath("Functional Pipeline")
    #     else:
    #         output_folder = PurePath(output_folder)
    #
    #     metadata_params = None
    #     if processed_dat_format.get(MetaTags.METATAG) is not None:
    #         metadata_params = processed_dat_format.get(MetaTags.METATAG)
    #         metadata_form = metadata_params.get(DataFormatType.METADATA)
    #
    #     acquisition = dict()
    #
    #     # Group files together based on location, modality, and video number
    #     # If we've selected modalities of interest, only process those; otherwise, process them all.
    #     if modes_of_interest is None:
    #         modes_of_interest = allData.loc[DataTags.MODALITY].unique().tolist()
    #
    #     for mode in modes_of_interest:
    #         modevids = allData.loc[allData[DataTags.MODALITY] == mode]
    #
    #         vidnums = np.unique(modevids[DataTags.VIDEO_ID].to_numpy())
    #         for num in vidnums:
    #             # Find the rows associated with this video number, and
    #             # extract the rows corresponding to this acquisition.
    #             acquisition = modevids.loc[modevids[DataTags.VIDEO_ID] == num]
    #
    #             im_info = acquisition.loc[acquisition[DataFormatType.FORMAT_TYPE] == DataFormatType.IMAGE]
    #             allData.loc[im_info.index, AcquisiTags.DATASET] = initialize_and_load_dataset(acquisition, metadata_params)
    #
    #
    #
    #     # Remove all entries without associated datasets.
    #     allData.drop(allData[allData[AcquisiTags.DATASET].isnull()].index, inplace=True)
    #
    #     grouping = pipeline_params.get(PipelineParams.GROUP_BY)
    #     if grouping is not None:
    #         for row in allData.itertuples():
    #             print( grouping.format_map(row._asdict()) )
    #             allData.loc[row.Index, PipelineParams.GROUP_BY] = grouping.format_map(row._asdict())
    #
    #         groups = allData[PipelineParams.GROUP_BY].unique().tolist()
    #     else:
    #         groups =[""] # If we don't have any groups, then just make the list an empty string.
    #
    #     for group in groups:
    #         if group != "":
    #             group_datasets = allData.loc[allData[PipelineParams.GROUP_BY] == group]
    #         else:
    #             group_datasets = allData
    #
    #         group_folder = output_folder.joinpath(group)
    #
    #         for m, mode in enumerate(modes_of_interest):
    #             ref_xforms=[]
    #
    #             modevids = group_datasets.loc[group_datasets[DataTags.MODALITY] == mode]
    #
    #             vidnums = modevids[DataTags.VIDEO_ID].to_numpy()
    #             datasets = modevids[AcquisiTags.DATASET].to_list()
    #             avg_images = np.dstack([data.avg_image_data for data in datasets])
    #
    #             if m == 0:
    #                 print("Selecting ideal central frame for mode and location: " + mode)
    #                 dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(avg_images),
    #                                                                             repeat(None),
    #                                                                             np.arange(len(datasets))))
    #                 shift_info = dist_res.get()
    #
    #                 # Determine the average
    #                 avg_loc_dist = np.zeros(len(shift_info))
    #                 f = 0
    #                 for allshifts in shift_info:
    #                     allshifts = np.stack(allshifts)
    #                     allshifts **= 2
    #                     allshifts = np.sum(allshifts, axis=1)
    #                     avg_loc_dist[f] = np.mean(np.sqrt(allshifts))  # Find the average distance to this reference.
    #                     f += 1
    #
    #                 print("Selecting ideal central frame for mode and location: " + mode)
    #
    #                 dist_res = pool.starmap_async(simple_image_stack_align, zip(repeat(avg_images),
    #                                                                             repeat(None),
    #                                                                             np.arange(len(datasets))))
    #                 shift_info = dist_res.get()
    #
    #                 # Determine the average
    #                 avg_loc_dist = np.zeros(len(shift_info))
    #                 f = 0
    #                 for allshifts in shift_info:
    #                     allshifts = np.stack(allshifts)
    #                     allshifts **= 2
    #                     allshifts = np.sum(allshifts, axis=1)
    #                     avg_loc_dist[f] = np.mean(np.sqrt(allshifts))  # Find the average distance to this reference.
    #                     f += 1
    #
    #                 avg_loc_idx = np.argsort(avg_loc_dist)
    #                 dist_ref_idx = avg_loc_idx[0]
    #
    #                 print("Determined most central dataset with video number: " + str(vidnums[dist_ref_idx]) + ".")
    #
    #                 central_dataset = datasets[dist_ref_idx]
    #
    #                 # Gaussian blur the data first before aligning, if requested
    #                 gausblur = pipeline_params.get(PipelineParams.GAUSSIAN_BLUR)
    #                 align_dat = avg_images.copy()
    #                 if gausblur is not None and gausblur != 0.0:
    #                     for f in range(avg_images.shape[-1]):
    #                         align_dat[..., f] = gaussian_filter(avg_images[..., f], sigma=gausblur)
    #
    #                 # Align the stack of average images from all datasets
    #                 align_dat, ref_xforms, inliers, avg_masks = optimizer_stack_align(align_dat,
    #                                                                                   (align_dat > 0),
    #                                                                                   dist_ref_idx,
    #                                                                                   determine_initial_shifts=True,
    #                                                                                   dropthresh=0.0, transformtype="affine")
    #
    #             # Apply the transforms to the unfiltered, cropped, etc. trimmed datasets, using the reference mode
    #             for f in range(avg_images.shape[-1]):
    #                 if inliers[f]:
    #                     avg_images[..., f] = cv2.warpAffine(avg_images[..., f], ref_xforms[f],
    #                                                         (avg_images.shape[1], avg_images.shape[0]),
    #                                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    #                                                         borderValue=np.nan)
    #
    #             # Z Project each of our image types
    #             avg_avg_images, avg_avg_mask = weighted_z_projection(avg_images)
    #
    #             # Determine the filename for the superaverage using the central-most dataset.
    #             pipelined_dat_format = dat_form.get("pipelined")
    #             if pipelined_dat_format is not None:
    #                 pipe_im_form = pipelined_dat_format.get(DataFormatType.IMAGE)
    #                 if pipe_im_form is not None:
    #                     pipe_im_fname = pipe_im_form.format_map(central_dataset.metadata)
    #
    #             # Make sure our output folder exists.
    #             central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder).mkdir(parents=True, exist_ok=True)
    #             cv2.imwrite(central_dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder, pipe_im_fname),
    #                         avg_avg_images)
    #
    #             print("Outputting data...")
    #             for dataset, xform in zip(datasets, ref_xforms):
    #                 # Make sure our output folder exists.
    #                 dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder).mkdir(parents=True, exist_ok=True)
    #
    #                 (rows, cols) = dataset.video_data.shape[0:2]
    #
    #                 if pipelined_dat_format is not None:
    #                     pipe_vid_form = pipelined_dat_format.get(DataFormatType.VIDEO)
    #                     pipe_mask_form = pipelined_dat_format.get(DataFormatType.MASK)
    #                     pipe_meta_form = pipelined_dat_format.get(MetaTags.METATAG)
    #
    #                     if pipe_vid_form is not None:
    #                         pipe_vid_fname = pipe_vid_form.format_map(dataset.metadata)
    #                     if pipe_mask_form is not None:
    #                         pipe_mask_fname = pipe_mask_form.format_map(dataset.metadata)
    #                     if pipe_meta_form is not None:
    #                         pipe_meta_form = pipe_meta_form.get(DataFormatType.METADATA)
    #                         if pipe_meta_form is not None:
    #                             pipe_meta_fname = pipe_meta_form.format_map(dataset.metadata)
    #
    #
    #                 og_dtype = dataset.video_data.dtype
    #                 for i in range(dataset.num_frames):  # Make all of the data in our dataset relative as well.
    #                     tmp = dataset.video_data[..., i].astype("float32")
    #                     tmp[np.round(tmp) == 0] = np.nan
    #                     tmp = cv2.warpAffine(tmp, xform,(cols, rows),
    #                                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    #                     tmp[np.isnan(tmp)] = 0
    #                     dataset.video_data[..., i] = tmp.astype(og_dtype)
    #
    #                     tmp = dataset.mask_data[..., i].astype("float32")
    #                     tmp[np.round(tmp) == 0] = np.nan
    #                     tmp = cv2.warpAffine(tmp, xform,(cols, rows),
    #                                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    #                     tmp[np.isnan(tmp)] = 0
    #                     dataset.mask_data[..., i] = tmp.astype(og_dtype)
    #
    #                 out_meta = pd.DataFrame(dataset.framestamps, columns=["FrameStamps"])
    #                 out_meta.to_csv(dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder, pipe_meta_fname), index=False)
    #                 save_video(dataset.metadata[AcquisiTags.BASE_PATH].joinpath(group_folder, pipe_vid_fname), dataset.video_data,
    #                            framerate=dataset.framerate)
    #
    #
    #
    # print("PK FIRE")