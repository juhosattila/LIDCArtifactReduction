# Operators here are using tensorflow in their implementation.

from LIDCArtifactReduction.tf_image import ssims_tf, mean_absolute_errors_tf, scale_Radio2HU, mean_squares_tf


def ssims_np(img1, img2):
    """Result is of shape (N,) or ()."""
    return ssims_tf(img1, img2).numpy()


def mean_absolute_errors_np(imgs1, imgs2):
    """Result is of shape (N,) or ()."""
    return mean_absolute_errors_tf(imgs1, imgs2).numpy()


def mean_absolute_errors_HU_np(imgs1, imgs2):
    """Result is of shape (N,) or ()."""
    return mean_absolute_errors_tf(
                scale_Radio2HU(imgs1),
                scale_Radio2HU(imgs2)).numpy()


def mean_squares_np(imgs):
    """Result is of shape (N,) or ()."""
    return mean_squares_tf(imgs)
