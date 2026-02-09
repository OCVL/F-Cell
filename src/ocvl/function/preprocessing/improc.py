import logging
import warnings

import cv2
import numpy as np
import SimpleITK as sitk
from colorama import Fore
from scipy import ndimage
from scipy.fft import next_fast_len, fft2, ifft2
from scipy.ndimage import binary_erosion
from numpy.polynomial import Polynomial

from tqdm import tqdm


def flat_field_frame(dataframe, mask=None, sigma=31, rescale=False):
    kernelsize = 3 * sigma
    if (kernelsize % 2) == 0:
        kernelsize += 1

    dataframe = dataframe.astype("float32")
    minval = np.nanmin(dataframe)
    maxval = np.nanmax(dataframe)

    if mask is None:
        mask = np.ones(dataframe.shape, dtype=dataframe.dtype)
        mask[dataframe == 0] = 0
    else:
        mask = mask.astype("float32")

    mask = ndimage.binary_closing(mask, np.ones((5,5))).astype("float32")

    #dataframe[dataframe == 0] = 0
    dataframe[np.isnan(dataframe)] = 0

    dataframe *= mask

    blurred_frame = cv2.GaussianBlur(dataframe, (kernelsize, kernelsize),
                                     sigmaX=sigma, sigmaY=sigma)
    blurred_mask = cv2.GaussianBlur(mask, (kernelsize, kernelsize),
                                     sigmaX=sigma, sigmaY=sigma)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="invalid value encountered in divide")

        blurred_frame /= blurred_mask

        blurred_frame /= np.nanmean(blurred_frame[:])

        flat_fielded = (dataframe / blurred_frame)
        flat_fielded[flat_fielded < minval] = minval
        flat_fielded[flat_fielded > maxval] = maxval

    mask[mask == 0] = np.nan
    flat_fielded *= mask

    return flat_fielded


def flat_field(dataset, mask=None, sigma=31, rescale=True):

    if mask is None:
        mask = dataset > 0

    if len(dataset.shape) > 2:
        flat_fielded_dataset = np.zeros(dataset.shape)
        for i in range(dataset.shape[-1]):
            flat_fielded_dataset[..., i] = flat_field_frame(dataset[..., i].copy(), mask[..., i].copy(), sigma, rescale)

        return flat_fielded_dataset
    else:
        return flat_field_frame(dataset.copy(), mask.copy(), sigma)


def norm_video(video_data, norm_method="mean", rescaled=False, rescale_mean=None, rescale_std=None):
    """
    This function normalizes a video (a single sample of all cells) using a method supplied by the user.

    :param video_data: A NxMxF numpy matrix with N rows, M columns, and F frames of some video.
    :param norm_method: The normalization method chosen by the user. Default is "mean". Options: "mean", "score", "median"
    :param rescaled: Whether or not to keep the data at the original scale (only modulate the numbers in place). Useful
                     if you want the data to stay in the same units. Default: False. Options: True/False
    :param rescale_mean: The mean scaling target for rescaling- if None, will use the mean of all data (excluding 0s)
    :param rescale_std:  The std dev scaling target for rescaling- if None, will use the std dev of all data (excluding 0s).
                        ignored in all scaling methods except "score"

    :return: a NxMxF numpy matrix of normalized video data.
    """
    logger = logging.getLogger("ORG_Logger")

    if norm_method == "mean":
        if rescale_mean is None:
            # Determine each frame's mean.
            flattened_vid = video_data.flatten().astype("float32")
            flattened_vid[flattened_vid == 0] = np.nan
            rescale_mean = np.nanmean(flattened_vid)
            del flattened_vid

        framewise_norm = np.empty([video_data.shape[-1]])
        framewise_std = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmean(frm)
            framewise_std[f]= np.nanstd(frm)

    elif norm_method == "score":
        if rescale_mean is None or rescale_std is None:
            # Determine each frame's mean.
            flattened_vid = video_data.flatten().astype("float32")
            flattened_vid[flattened_vid == 0] = np.nan
            rescale_mean = np.nanmean(flattened_vid)
            rescale_std = np.nanstd(flattened_vid)
            del flattened_vid

        framewise_norm = np.empty([video_data.shape[-1]])
        framewise_std = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmean(frm)
            framewise_std[f] = np.nanstd(frm)

        # Standardizes the data into a zscore- then rescales it to a common std dev and mean
        rescaled_vid = np.zeros(video_data.shape, dtype=np.float32)
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].astype("float32")
            frm[frm == 0] = np.nan # This is to prevent bad behavior when we do subtraction of the mean- e.g:
                                   # 0 will become -framewise_norm, cats and dogs will live together; pandemonium.
            # frm = np.log(frm)
            rescaled_vid[:, :, f] = (frm - framewise_norm[f]) / framewise_std[f]



    elif norm_method == "median":
        # Determine each frame's median.
        framewise_norm = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmedian(frm)
        all_norm = np.nanmean(framewise_norm)
    elif norm_method == "flatfield":
        if rescaled:
            logger.warning("Flat-field based video normalization ignores the \"rescaled\" parameter.")
        return flat_field(video_data)
    else:
        # Determine each frame's mean.
        framewise_norm = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmean(frm)
        all_norm = np.nanmean(framewise_norm)
        logger.warning("The \"" + norm_method + "\" normalization type is not recognized. Defaulting to mean.")


    if rescaled: # Provide the option to simply scale the data, instead of keeping it in relative terms
        if norm_method != "score":
            ratio = framewise_norm / rescale_mean
            rescaled_vid = np.empty(video_data.shape, dtype=np.float32)
            for f in range(video_data.shape[-1]):
                rescaled_vid[:, :, f] = video_data[:, :, f].astype("float32") / ratio[f]
        else:
            rescaled_vid = ((rescaled_vid * rescale_std) + rescale_mean).astype("float32")
    else:
        rescaled_vid = np.zeros(video_data.shape, dtype=np.float32)
        for f in range(video_data.shape[-1]):
            rescaled_vid[:, :, f] = video_data[:, :, f].astype("float32") / framewise_norm[f]

    # Used to validate outputs.
    # reframewise_norm = np.empty([video_data.shape[-1]])
    # reframewise_std = np.empty([video_data.shape[-1]])
    # for f in range(video_data.shape[-1]):
    #     frm = rescaled_vid[:, :, f].flatten().astype("float32")
    #     frm[frm == 0] = np.nan
    #     reframewise_norm[f] = np.nanmean(frm)
    #     reframewise_std[f] = np.nanstd(frm)
    #save_tiff_stack("std_vid.tif", rescaled_vid)
    return rescaled_vid




def dewarp_2D_data(image_data, row_shifts, col_shifts, method="median", fitshifts = False):
    '''
    # Where the image data is N rows x M cols and F frames
    # and the row_shifts and col_shifts are F x N.
    # Assumes a row-wise distortion/a row-wise fast scan ("distortionless" along each row)
    # Returns a float image (spans from 0-1).
    :param fitshifts:
    :param image_data:
    :param row_shifts:
    :param col_shifts:
    :param method:
    :return:
    '''
    numstrips = row_shifts.shape[1]
    height = image_data.shape[0]
    width = image_data.shape[1]
    num_frames = image_data.shape[-1]

    allrows = np.linspace(0, numstrips - 1, num=height)  # Make a linspace for all of our images' rows.
    substrip = np.linspace(0, numstrips - 1, num=numstrips)


    indiv_colshift = np.zeros([num_frames, height])
    indiv_rowshift = np.zeros([num_frames, height])

    if fitshifts:
        for f in range(num_frames):
            # Fit across rows, in order to capture all strips for a given dataset
            finite = np.isfinite(col_shifts[f, :])
            col_strip_fit = Polynomial.fit(substrip[finite], col_shifts[f, finite], deg=12)
            indiv_colshift[f, :] = col_strip_fit(allrows)
            # Fit across rows, in order to capture all strips for a given dataset
            finite = np.isfinite(row_shifts[f, :])
            row_strip_fit = Polynomial.fit(substrip[finite], row_shifts[f, finite], deg=12)
            indiv_rowshift[f, :] = row_strip_fit(allrows)
    else:
        indiv_colshift = col_shifts
        indiv_rowshift = row_shifts

    if method == "median":
        centered_col_shifts = -np.nanmedian(indiv_colshift, axis=0)
        centered_row_shifts = -np.nanmedian(indiv_rowshift, axis=0)

    dewarped = np.zeros(image_data.shape, dtype=np.float32)

    col_base = np.tile(np.arange(width, dtype=np.float32)[np.newaxis, :], [height, 1])
    row_base = np.tile(np.arange(height, dtype=np.float32)[:, np.newaxis], [1, width])

    centered_col_shifts = col_base + np.tile(centered_col_shifts[:, np.newaxis], [1, width]).astype("float32")
    centered_row_shifts = row_base + np.tile(centered_row_shifts[:, np.newaxis], [1, width]).astype("float32")

    premask_dtype = image_data.dtype

    if premask_dtype == np.float32 or premask_dtype == np.float64:
        datmax = 1
    else:
        datmax = np.iinfo(premask_dtype).max

    for f in range(num_frames):
        norm_frame = image_data[..., f].astype("float64") / datmax
        norm_frame[norm_frame == 0] = np.nan
        dewarped[..., f] = cv2.remap(norm_frame,
                                     centered_col_shifts,
                                     centered_row_shifts,
                                     interpolation=cv2.INTER_LINEAR)

    # Clamp our values, and convert nan to 0s to maintain compatability
    dewarped[dewarped < 0] = 0
    dewarped[np.isnan(dewarped)] = 0
    dewarped[dewarped > 1] = 1

    return (dewarped*datmax).astype(premask_dtype), centered_col_shifts, centered_row_shifts




# Calculate a running sum in all four directions - apparently more memory efficient
# Used for determining the total energy at every point that an image contains.
def local_sum(matrix, overlap_shape):

    if matrix.shape[0] == overlap_shape[0] and matrix.shape[1] == overlap_shape[1]:
        energy_mat = np.cumsum(matrix, axis=0)
        energy_mat = np.pad(energy_mat, ((0, energy_mat.shape[0]-1), (0, 0)),  mode="reflect") # This is wrong- needs to be a pad with end value
                # and a subtraction of the above nrg matrix. Not sure why intuitively. Maybe FD wrapping?
        energy_mat = np.cumsum(energy_mat, axis=1)
        energy_mat = np.pad(energy_mat, ((0, 0), (0, energy_mat.shape[1]-1)), mode="reflect")

    else:
        print("This code does not yet support unequal image sizes!")

    return energy_mat


# Using Dirk Padfield's Masked FFT Registration approach (Normalized xcorr for arbitrary masks).
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5540032
def general_normxcorr2(template_im, reference_im, template_mask=None, reference_mask=None, required_overlap=None):
    temp_size = template_im.shape
    ref_size = reference_im.shape

    template = template_im.astype("float32")
    reference = reference_im.astype("float32")

    template[np.isnan(template)] = 0
    reference[np.isnan(reference)] = 0

    # Speed up FFT by padding to optimal size.
    ogrows = temp_size[0] + ref_size[0] - 1
    ogcols = temp_size[1] + ref_size[1] - 1
    target_size = (next_fast_len(ogrows), next_fast_len(ogcols))

    if template_mask is None:
        template_mask = template_im > 0
    if reference_mask is None:
        reference_mask = reference_im > 0

    # First, cross correlate our two images (but this isn't normalized, yet!)
    # The templates should be rotated by 90 degrees. So...
    template_mask = np.rot90(template_mask.copy(), k=2)
    template = np.rot90(template, k=2)

    f_one = fft2(reference, target_size)
    f_one_sq = fft2(reference*reference, target_size)
    m_one = fft2(reference_mask, target_size)

    f_two = fft2(template, target_size)
    f_two_sq = fft2(template * template, target_size)
    m_two = fft2(template_mask, target_size)

    base_xcorr = ifft2(f_one * f_two).real

    # Fulfill equations 10-12 from the paper.
    # First get the overlapping energy...
    pixelwise_overlap = ifft2(m_one * m_two).real  # Eq 10
    pixelwise_overlap[pixelwise_overlap <= 0] = 1

    # For the template frame denominator portion.
    ref_corrw_one = ifft2(f_one * m_two).real  # Eq 11
    ref_sq_corrw_one = ifft2(f_one_sq * m_two).real  # Eq 12

    ref_denom = ref_sq_corrw_one - ((ref_corrw_one * ref_corrw_one) / pixelwise_overlap)
    ref_denom[ref_denom < 0] = 0  # Clamp these values to 0.

    # For the reference frame denominator portion.
    temp_corrw_one = ifft2(m_one * f_two).real  # Eq 11
    temp_sq_corrw_one = ifft2(m_one * f_two_sq).real  # Eq 12

    temp_denom = temp_sq_corrw_one - ((temp_corrw_one * temp_corrw_one) / pixelwise_overlap)
    temp_denom[temp_denom < 0] = 0  # Clamp these values to 0.

    # Construct our numerator
    numerator = base_xcorr - ((ref_corrw_one*temp_corrw_one)/pixelwise_overlap)
    denom = np.sqrt(temp_denom)*np.sqrt(ref_denom)

    # Need this bit to avoid dividing by zero.
    tolerance = 1000*np.finfo(np.amax(denom)).eps

    xcorr_out = numerator / (denom + 1)

    # By default, the images have to overlap by more than 20% of their maximal overlap.
    if required_overlap is None:
        required_overlap = np.amax(pixelwise_overlap)*0.5
    else:
        required_overlap = np.amax(pixelwise_overlap)*required_overlap

    xcorr_out[pixelwise_overlap < required_overlap ] = 0

    maxval = np.amax(xcorr_out[:])
    maxloc = np.unravel_index(np.argmax(xcorr_out[:]), xcorr_out.shape)
    maxshift = (-float(maxloc[1]-np.floor(ogcols/2.0)), -float(maxloc[0]-np.floor(ogrows/2.0))) #Output as X and Y.
    #pyplot.imshow(xcorr_out, cmap='gray')
    #pyplot.show()

    return maxshift, maxval, xcorr_out


def simple_image_stack_align(im_stack, mask_stack=None, ref_idx=0, overlap =0.5):
    logger = logging.getLogger("ORG_Logger")
    num_frames = im_stack.shape[-1]
    shifts = [None] * num_frames
    # flattened = flat_field(im_stack)
    flattened = (im_stack)

    logger.info(f"Performing a NCC-based alignment to frame #{ref_idx}")
    pbar = tqdm(range(num_frames), bar_format=("%s{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN, Fore.GREEN)), unit="frm")

    if mask_stack is not None:
        for f2 in pbar:
            pbar.set_description(f"Aligning frame {f2} of {num_frames}")
            shift, val, xcorrmap = general_normxcorr2(flattened[..., f2], flattened[..., ref_idx],
                                                      template_mask=mask_stack[..., f2],
                                                      reference_mask=mask_stack[..., ref_idx],
                                                      required_overlap=overlap)
            #print("Found shift of: " + str(shift) + ", value of " + str(val))
            shifts[f2] = shift
    else:
        for f2 in range(0, num_frames):
            shift, val, xcorrmap = general_normxcorr2(flattened[..., f2], flattened[..., ref_idx], required_overlap=overlap)
            #print("Found shift of: " + str(shift) + ", value of " + str(val))
            shifts[f2] = shift

    return shifts

def simple_dataset_list_align(datasets, ref_idx):
    num_frames = len(datasets)
    shifts = [None] * num_frames

    ref_dataset = datasets[ref_idx]
    print("Aligning to dataset with video number: " + str(ref_idx))
    i = 0
    for dataset in datasets:
        shift, val, xcorrmap = general_normxcorr2(dataset.avg_image_data, ref_dataset.avg_image_data)
        #print("Found shift of: " + str(shift) + ", value of " + str(val))
        shifts[i] = shift
        i+=1

    return shifts


def optimizer_stack_align(im_stack, mask_stack, reference_idx, determine_initial_shifts=True, dropthresh=None, transformtype="affine", justalign=False):
    logger = logging.getLogger("ORG_Logger")

    num_frames = im_stack.shape[-1]
    og_dtype = im_stack.dtype
    if not justalign:
        reg_stack = np.zeros(im_stack.shape)
        reg_mask = np.zeros(mask_stack.shape)
    else:
        reg_stack = None
        reg_mask = None

    eroded_mask = np.zeros(mask_stack.shape, dtype=mask_stack.dtype)

    # Erode our masks a bit to help with stability.
    logger.info(f"Refining the {num_frames} frame mask stack... ")
    pbar = tqdm(range(num_frames), bar_format=("%s{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN, Fore.GREEN)), unit="frm")

    for f in pbar:
        pbar.set_description(f"Refining frame {f} of {num_frames}")

        eroded_mask[..., f] = binary_erosion(mask_stack[..., f], structure=np.ones((21, 21)))

    if determine_initial_shifts:
        initial_shifts = simple_image_stack_align(im_stack * eroded_mask, eroded_mask, reference_idx)
    else:
        initial_shifts = [(0.0, 0.0)] * num_frames


    imreg_method = sitk.ImageRegistrationMethod()

    imreg_method.SetMetricAsCorrelation()
    imreg_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.0625, minStep=1e-5,
                                                          numberOfIterations=500,
                                                          relaxationFactor=0.6, gradientMagnitudeTolerance=1e-5)
    imreg_method.SetOptimizerScalesFromPhysicalShift() #This apparently allows parameters to change independently of one another.
                                                      # And is incredibly important.
    # #https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/61_Registration_Introduction_Continued.html#Final-registration

    # im_stack = im_stack.astype("float32")
    #
    # im_stack[np.isnan(im_stack)] = 0

    ref_im = sitk.GetImageFromArray(im_stack[..., reference_idx].astype("float32"))
    #ref_im = sitk.Cast(ref_im, sitk.sitkfloat32)
    #ref_im = sitk.Normalize(ref_im)
    dims = ref_im.GetDimension()

    imreg_method.SetMetricFixedMask(sitk.GetImageFromArray(eroded_mask[..., reference_idx].astype("float32")))

    xforms = [None] * num_frames
    inliers = np.zeros(num_frames, dtype=bool)
    logger.info(f"Aligning {num_frames} frame stack... ")
    pbar = tqdm(range(num_frames), bar_format=("%s{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN, Fore.GREEN)), unit="frm")

    for f in pbar:
        pbar.set_description( f"Aligning frame {f} of {num_frames}")

        if transformtype == "rigid":
            xForm = sitk.Euler2DTransform()
        elif transformtype == "affine":
            xForm = sitk.AffineTransform(2)

        xForm.SetCenter((np.array(im_stack[..., reference_idx].shape, dtype="float")-1) / 2)
        xForm.SetTranslation(initial_shifts[f])

        imreg_method.SetInitialTransform(xForm)
        imreg_method.SetInterpolator(sitk.sitkLinear)

        moving_im = sitk.GetImageFromArray(im_stack[..., f].astype("float32"))
        #moving_im = sitk.Normalize(moving_im)
        imreg_method.SetMetricMovingMask(sitk.GetImageFromArray(eroded_mask[..., f].astype("float32")))

        outXform = imreg_method.Execute(ref_im, moving_im)

        if dropthresh is not None and imreg_method.GetMetricValue() > -dropthresh:
            inliers[f] = False
        else:
            inliers[f] = True

        if transformtype == "rigid":
            outXform = sitk.Euler2DTransform(outXform)
        elif transformtype == "affine":
            outXform = sitk.AffineTransform(outXform)

        # ITK's "true" transforms are found as follows: T(x)=A(x−c)+t+c
        A = np.array(outXform.GetMatrix()).reshape(2, 2)
        c = np.array(outXform.GetCenter())
        t = np.array(outXform.GetTranslation())

        Tx = np.eye(3)
        Tx[:2, :2] = A
        Tx[0:2, 2] = -np.dot(A, c)+t+c
        xforms[f] = Tx[0:2, :]

        if inliers[f] and not justalign:
            norm_frame = im_stack[..., f].astype("float32")
            # Make all masked data nan so that when we transform them we don't have weird edge effects
            norm_frame[mask_stack[..., f] == 0] = np.nan

            norm_frame = cv2.warpAffine(norm_frame, xforms[f],(norm_frame.shape[1], norm_frame.shape[0]),
                                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=np.nan)

            reg_mask[..., f] = np.isfinite(norm_frame).astype(og_dtype) # Our new mask corresponds to the real data.
            norm_frame[np.isnan(norm_frame)] = 0 # Make anything that was nan into a 0, to be kind to non nan-types
            reg_stack[..., f] = norm_frame

    print("Done.")

    if justalign:
        return None, xforms, inliers, None
    else:
        return reg_stack.astype(og_dtype), xforms, inliers, reg_mask.astype(og_dtype)
     # save_video(
     #     "B:\\Dropbox\\testalign.avi",
     #      reg_stack, 25)



def relativize_image_stack(image_data, mask_data=None, reference_idx=0, numkeypoints=10000, method="affine", dropthresh=None):

    if mask_data is None:
        mask_data = (image_data > 0).astype(image_data.dtype)
    num_frames = image_data.shape[-1]

    xform = [None] * num_frames
    corrcoeff = np.empty((num_frames, 1))
    corrcoeff[:] = np.nan
    corrected_stk = np.zeros(image_data.shape)

    sift = cv2.SIFT_create(numkeypoints, nOctaveLayers=55, contrastThreshold=0.0, sigma=1)

    keypoints = []
    descriptors = []

    for f in range(num_frames):
        kp, des = sift.detectAndCompute(image_data[..., f], mask_data[..., f], None)
        # if numkeypoints > 8000:
        #     print("Found "+ str(len(kp)) + " keypoints")
        # Normalize the features by L1; (make this RootSIFT) instead.
        des /= (des.sum(axis=1, keepdims=True) + np.finfo(float).eps)
        des = np.sqrt(des)
        keypoints.append(kp)
        descriptors.append(des)


    # Set up FLANN parameters (feature matching)... review these.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)

    matcher = cv2.BFMatcher.create()

    # Specify the number of iterations.
    for f in range(num_frames):

        matches = matcher.knnMatch(descriptors[f], descriptors[reference_idx], k=2)

        good_matches = []
        for f1, f2 in matches:
            if f1.distance < 0.7 * f2.distance:
                good_matches.append(f1)

        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints[f][f1.queryIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[reference_idx][f1.trainIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)

            if f != reference_idx:
                img_matches = np.empty((max(image_data[..., f].shape[0], image_data[..., f].shape[0]), image_data[..., f].shape[1] + image_data[..., f].shape[1], 3),
                                       dtype=np.uint8)

                cv2.drawMatches( image_data[..., f], keypoints[f], image_data[..., reference_idx], keypoints[reference_idx], good_matches, img_matches)
                cv2.imshow("meh", img_matches)
                cv2.waitKey()

            if method == "affine":
                M, inliers = cv2.estimateAffine2D(dst_pts, src_pts) # More stable- also means we have to set the inverse flag below.
            else:
                M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)

            if M is not None and np.sum(inliers) >= 4:
                xform[f] = M+0

                corrected_stk[..., f] = cv2.warpAffine(image_data[..., f], xform[f], np.flip(image_data[..., f].shape),
                                                      flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                warped_mask = cv2.warpAffine(mask_data[..., f], xform[f], np.flip(mask_data[..., f].shape),
                                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

                # Calculate and store the final correlation. It should be decent, if the transform was.
                res = cv2.matchTemplate(image_data[..., reference_idx], corrected_stk[..., f].astype("uint8"),
                                    cv2.TM_CCOEFF_NORMED, mask=warped_mask)

                corrcoeff[f] = res.max()

                # print("Found " + str(np.sum(inliers)) + " matches between frame " + str(f) + " and the reference, for a"
                #                                         " normalized correlation of " + str(corrcoeff[f]))
            else:
                pass
                #print("Not enough inliers were found: " + str(np.sum(inliers)))
        else:
            print("Not enough matches were found: " + str(len(good_matches)))

    if dropthresh is None:
        print("No drop threshold detected, auto-generating...")
        dropthresh = np.nanquantile(corrcoeff, 0.01)


    corrcoeff[np.isnan(corrcoeff)] = 0  # Make all nans into zero for easy tracking.

    inliers = np.squeeze(corrcoeff >= dropthresh)
    corrected_stk = corrected_stk[..., inliers]
    # save_video(
    #     "\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\test_corrected_stk.avi",
    #     corrected_stk, 29.4)
    for i in range(len(inliers)):
        if not inliers[i]:
            xform[i] = None # If we drop a frame, eradicate its xform. It's meaningless anyway.

    print("Using a threshold of "+ str(dropthresh) +", we kept " + str(np.sum(corrcoeff >= dropthresh)) + " frames. (of " + str(num_frames) + ")")


    return corrected_stk, xform, inliers


def weighted_z_projection(image_data, weights=None, projection_axis=-1, type="average"):
    num_frames = image_data.shape[-1]

    og_dtype = image_data.dtype
    if weights is None:
        weights = image_data > 0

    # image_data = image_data.astype("float32")
    # weights = weights.astype("float32")

    maxstart = np.nanmax(image_data.flatten())
    minstart = np.nanmin(image_data.flatten())

    image_projection = image_data * weights
    image_projection = np.nansum(image_projection, axis=projection_axis).astype("float64")
    weight_projection = np.nansum(weights, axis=projection_axis).astype("float64")
    weight_projection[weight_projection == 0] = np.nan

    image_projection /= weight_projection

    weight_projection[np.isnan(weight_projection)] = 0
    image_projection[np.isnan(image_projection)] = 0
    image_projection[image_projection < 0] = 0

    if og_dtype == np.uint8:
        image_projection[image_projection > 255] = 255
        image_projection=image_projection.astype("uint8")
    else:
        image_projection[image_projection > maxstart] = maxstart
        image_projection[image_projection < minstart] = minstart

    # pyplot.imshow(image_projection, cmap='gray')
    # pyplot.show()
    # return image_projection.astype("uint8"), (weight_projection / np.nanmax(weight_projection.flatten()))

    return image_projection.astype(og_dtype), (weight_projection / np.nanmax(weight_projection.flatten()))
