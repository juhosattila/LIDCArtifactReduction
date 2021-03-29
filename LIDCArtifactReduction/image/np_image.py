# Operators here may use tensorflow in their implementation.
import numpy as np
from skimage import measure

from LIDCArtifactReduction.image.tf_image import ssims_tf, mean_absolute_errors_tf, scale_Radiodiff2HUdiff, mean_squares_tf, \
    scale_Radio2HU


def ssims_np(img1, img2):
    """Result is of shape (N,) or ()."""
    return ssims_tf(img1, img2).numpy()


def mean_absolute_errors_np(imgs1, imgs2):
    """Result is of shape (N,) or ()."""
    return mean_absolute_errors_tf(imgs1, imgs2).numpy()


def mean_absolute_errors_HU_np(imgs1, imgs2):
    """Result is of shape (N,) or ()."""
    return mean_absolute_errors_tf(
                scale_Radiodiff2HUdiff(imgs1),
                scale_Radiodiff2HUdiff(imgs2)).numpy()


def mean_squares_np(imgs):
    """Result is of shape (N,) or ()."""
    return mean_squares_tf(imgs).numpy()


#### ================================================================================#####
# TODO: think if these fit here. They do not use tensorflow in their implementation.

def segment_lung(img, return_padded=False, return_thresholded_img=False):
    """Works in volume, i.e. NHW, or slice, i.e. HW, modes. Channel dimension is forbidden."""
    rank = np.ndim(img)
    thresholded_img = np.array(scale_Radio2HU(img) > -500, dtype=np.int)


    if rank == 2:
        padded_thresholded_img = np.pad(thresholded_img, pad_width=((1, 1), (1, 1)), constant_values=0)
    else:  # rank == 3
        padded_thresholded_img = np.pad(thresholded_img, pad_width=((0, 0), (1, 1), (1, 1)), constant_values=0)

    padded_labeled_img = measure.label(padded_thresholded_img, background=-1)
    if rank == 2:
        background_air_label = padded_labeled_img[0, 0]
    else:  # rank == 3
        background_air_label = padded_labeled_img[0, 0, 0]

    padded_seg_lung_img = 1 - np.where(padded_labeled_img == background_air_label, 1, padded_thresholded_img)

    if return_padded:
        return_value = padded_seg_lung_img
    else:
        if rank == 2:
            return_value = padded_seg_lung_img[1:-1, 1:-1]
        else:  # rank == 3
            return_value = padded_seg_lung_img[:, 1:-1, 1:-1]

    if return_thresholded_img:
        return_value = (return_value, thresholded_img)

    return return_value


def largest_label_region(labeled):
    vals, counts = np.unique(labeled, return_counts=True)
    counts = counts[vals != 0]
    vals = vals[vals != 0]

    if len(counts) == 0:
        return None

    return vals[np.argmax(counts)]


def _segment_lung_bronchial2D(img):
    """Does not really work on 3D data, because in 3D bronchials are connected to the torso."""
    seg_lung_img = segment_lung(img)
    labeled = measure.label(seg_lung_img, background=1)
    lmax = largest_label_region(labeled)
    seg_lung_bronchial_img = np.where(labeled != lmax, 1, 0)
    return seg_lung_bronchial_img


def segment_lung_bronchial(img):
    """Works in volume, i.e. NHW, or slice, i.e. HW, modes. Channel dimension is forbidden.
    Only call for connected volume and do not stack non-connected slices from different patients."""
    if np.ndim(img) == 2:
        return _segment_lung_bronchial2D(img)

    else:  # ndim = 3
        return np.stack([_segment_lung_bronchial2D(i) for i in img], axis=0)


def segment_lung_bronchial_torso(img, return_lung=False):
    seg_lung_img, thresholded_img = segment_lung(img, return_thresholded_img=True)
    torso_desk = seg_lung_img | thresholded_img
    labeled = measure.label(torso_desk, background=0)
    lmax = largest_label_region(labeled)
    seg_lung_bronchial_torso_img = np.where(labeled == lmax, 1, 0)

    if return_lung:
        return seg_lung_bronchial_torso_img, seg_lung_img

    return seg_lung_bronchial_torso_img


def segment_bronchial_torso(img):
    seg_lbt_img, seg_lung_img = segment_lung_bronchial_torso(img, return_lung=True)
    return seg_lbt_img - seg_lung_img

#
# def overlap_segmentation(img, seg_binint):
#     imgc = np.repeat(np.expand_dims(scale_HU2Radio(img), axis=-1), 3, axis=-1)
#     seg_binintc = np.stack([seg_binint, np.zeros_like(seg_binint), np.zeros_like(seg_binint)], axis=-1)
#     return (imgc + seg_binintc) / 2.5
