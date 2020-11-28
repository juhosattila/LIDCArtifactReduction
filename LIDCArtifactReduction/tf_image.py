from tensorflow_addons.utils.types import TensorLike

import tensorflow as tf
import tensorflow_addons as tfa

from LIDCArtifactReduction import parameters


def _get_scale_angle_translate_centre_matrix(scale, angle, translate, centre):
    """Creates a 2D transformation in homogenoues coordinates based on the formula:
    y=S(R(mean-c0) + t + c0)

    Translate to origin, rotate, apply arbitrary translation, translate back to centre.
    There is additional scaling for the sake of resampling.

    Args:
        scale: either a constant or a matrix
    """
    # R
    rotation_matrix = tf.reshape(tf.convert_to_tensor(
        [tf.cos(angle), -tf.sin(angle),
         tf.sin(angle), tf.cos(angle)]
    ), shape=[2, 2])

    real_scale = scale
    if tf.rank(real_scale) >= tf.constant(1):
        real_scale = tf.linalg.diag(real_scale)

    scaled_rotation_matrix = real_scale * rotation_matrix

    # s*(-R*c0 + t + c0)
    scaled_translation = real_scale * (-tf.linalg.matvec(rotation_matrix, centre) + translate + centre)

    # [ sR | s*(-R*c0 + t + c0)]
    # [0 0 |  1 ]
    transformation_matix = tf.reshape(tf.convert_to_tensor(
        [scaled_rotation_matrix[0, 0], scaled_rotation_matrix[0, 1], scaled_translation[0],
         scaled_rotation_matrix[1, 0], scaled_rotation_matrix[1, 1], scaled_translation[1],
         0., 0., 1.]
    ), shape=[3, 3])

    # inverted
    return tf.linalg.inv(transformation_matix)



def apply_rotate_translate_resampling(input_img: tf.Tensor, angle: float,
                                      translate: tf.Tensor = tf.constant([0., 0.]),
                                      output_img_side_length: tf.Tensor = None):
    """Apply a rotation and translation to an image with a possible new dimension.

    Assume image is square. Output image will also be a square.

    Args:
         output_img_side_length: it is a Tensor wrapping a single integer number.

    Example:
        transformed = apply_rotate_translate_resampling(
            input_img=img_sample_tf, angle=math.pi/6,
            translate=tf.constant([256.0, 0.0]), output_img_side_length=tf.constant(1024))
    """
    _angle = tf.convert_to_tensor(angle, dtype=tf.float32)
    input_img_side_length = tf.shape(input_img)[0]
    real_output_img_side_length = output_img_side_length \
        if output_img_side_length is not None else input_img_side_length
    transformation_parameters = get_rotate_translate_resampling_params(input_img_side_length,
                                                                       _angle, translate, real_output_img_side_length)
    return tfa.image.transform(images=input_img,
                               transforms=transformation_parameters,
                               interpolation='BILINEAR',
                               output_shape=[real_output_img_side_length, real_output_img_side_length])


def get_rotate_translate_resampling_params(input_img_side_length, angle,
                                           translate=tf.constant([0., 0.]), output_img_side_length=None):
    """Returns a tf.image.transform compliant 2D transformation (8 element array) for rotation, translation.

    [input_img_side_length/2, input_img_side_length/2] represents the centre of the transformation.
    Based on output_img_side_length we have scaling.
    It inverts the transformation.
    Assume, images are square.
    """
    real_output_img_side_length = output_img_side_length \
        if output_img_side_length is not None else input_img_side_length
    scale = tf.cast(real_output_img_side_length, dtype=tf.float32) / tf.cast(input_img_side_length, dtype=tf.float32)
    centre = tf.convert_to_tensor([input_img_side_length, input_img_side_length], dtype=tf.float32) * 0.5

    return tf.reshape(_get_scale_angle_translate_centre_matrix(scale, angle, translate, centre), [9])[:-1]


def min_max_scale(img_or_imgs: TensorLike):
    """Scale in a min-max fashion a list of Tensors to interval [0, 1].

    Args:
        img_or_imgs: a NHWC or HWC mode tensor.
    """
    img_or_imgs = tf.convert_to_tensor(img_or_imgs, dtype=tf.float32)

    per_image_min = tf.reduce_min(img_or_imgs, axis=[-3, -2, -1])
    per_image_max = tf.reduce_max(img_or_imgs, axis=[-3, -2, -1])
    per_image_diff = per_image_max - per_image_min

    broadcastable_shape = tf.concat([tf.shape(per_image_diff), [1, 1, 1]], axis=0)

    diff_ex = tf.reshape(per_image_diff, shape=broadcastable_shape)
    min_ex = tf.reshape(per_image_min, shape=broadcastable_shape)

    return (img_or_imgs - min_ex) / diff_ex


# HU should be understood as difference measured in HU and not absolute scales.
# HU is not translated necessarily anywhere, but rescaling should be applied to reach
# to radiosities and absolute gray levels.
#
# Radiosity measure is custom rescaling created for own purposes. We use HU_TO_CT_SCALING as scaling parameter.
#
# Absolute gray levels is a concept found in dicom files. Interceps and slopes give the transformation to an absolute
# HU stddev from where we reach radiosity levels.


def scale_HU2Radio(imgs: TensorLike):
    img_min = tf.constant([0.], dtype=tf.float32)
    img_max = tf.constant([parameters.HU_TO_CT_SCALING], dtype=tf.float32)
    return (imgs - img_min) / (img_max - img_min)


def scale_Radio2HU(imgs: TensorLike):
    img_min = tf.constant([0.], dtype=tf.float32)
    img_max = tf.constant([parameters.HU_TO_CT_SCALING], dtype=tf.float32)
    return imgs * (img_max - img_min) + img_min


def scale_Gray2Radio(imgs: TensorLike, intercepts, slopes):
    """Scale an absolute grey level image to linear attenuation coefficient space, based on intercepts and slopes.

    Air is considered to have level 0. Bone is considered to have value 1000.

    Args:
        imgs: Tensor in NHWC or HWC format.
    """
    imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)

    def expand_dim_per_image_data(per_image_data: TensorLike):
        new_shape = tf.concat([tf.shape(per_image_data), [1, 1, 1]], axis=0)
        return tf.reshape(per_image_data, shape=new_shape)

    intercepts_tf = tf.convert_to_tensor(intercepts, dtype=tf.float32)
    intercepts_tf = expand_dim_per_image_data(intercepts_tf)

    slopes_tf = tf.convert_to_tensor(slopes, dtype=tf.float32)
    slopes_tf = expand_dim_per_image_data(slopes_tf)

    bias_tf = tf.constant([1024.], dtype=tf.float32)

    # Without adding the bias we get absolute HU values.
    rescaled_tf = imgs_tf * slopes_tf + intercepts_tf + bias_tf

    # scaling of (min,1000) to (0,1)
    # TODO: do it based on minimum, and not 0
    zeroed_out_tf = tf.where(rescaled_tf >= 0, rescaled_tf, tf.zeros_like(rescaled_tf))

    return scale_HU2Radio(zeroed_out_tf)


def total_variation_op(imgs: TensorLike, eps=0.0):
    eps_tf = tf.convert_to_tensor(eps, dtype=tf.float32)
    imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
    imgs4D = tf.reshape(imgs_tf, shape=tf.concat([tf.shape(imgs_tf)[:3], [1]], axis=0))
    diff_x = imgs4D[:, :, 1:] - imgs4D[:, :, :-1]
    diff_y = imgs4D[:, 1:] - imgs4D[:, :-1]
    total_variation = tf.sqrt(tf.square(diff_x[:, :-1]) + tf.square(diff_y[:, :, :-1]) + eps_tf)
    return total_variation


def total_variation_sum_norm(imgs: TensorLike):
    """Basically TV-L1.

    Args:
         imgs: Should be a 3D NHW or 4D NHWC tensor with C=1.
    """
    total_variation = total_variation_op(imgs)
    tv_norm = tf.reduce_sum(total_variation, axis=[1, 2, 3])

    # For testing
    # return tv_norm, total_variation

    return tv_norm


def total_variation_mean_norm(imgs: TensorLike):
    """Normalised TV-L1 norm.

    Args:
         imgs: Should be a 3D NHW or 4D NHWC tensor with C=1.
    """
    total_variation = total_variation_op(imgs)
    tv_norm = tf.reduce_mean(total_variation, axis=[1, 2, 3])

    # For testing.
    # return tv_norm, total_variation

    return tv_norm


class TotalVariationNormObjectiveFunction:
    def __call__(self, imgs: TensorLike):
        return total_variation_mean_norm(imgs)


def reweighted_total_variation_norm(imgs: TensorLike, delta : float):
    """Get reweighted total variation norm of imgs.

    Args:
        imgs: Should be a 3D NHW or 4D NHWC tensor with C=1.
        delta: positive float

    Returns:
        1D tensor of reweighted total variation norms.
    """
    total_variation = total_variation_op(imgs)
    reweighted_tv = total_variation / (total_variation + delta)

    reweighted_tv_norm = tf.reduce_sum(reweighted_tv, axis=[1, 2, 3])

    # For testing.
    # return reweighted_tv_norm, reweighted_tv

    return reweighted_tv_norm


def sparsity_sum_operator(imgs: TensorLike, eps: float):
    return tf.reduce_sum(tf.math.log(imgs + eps))


def sparsity_mean_operator(imgs: TensorLike, eps: float):
    return tf.reduce_mean(tf.math.log(imgs + eps))


def sparse_total_variation_objective_function(imgs: TensorLike, eps: float):
    # Eps is used in TV gradients, because derivatives were becoming 0/0 after a while.
    # Think about reducing eps.
    total_variation = total_variation_op(imgs, eps=1e-3)
    return sparsity_mean_operator(total_variation, eps)


class SparseTotalVariationObjectiveFunction:
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, imgs):
        return sparse_total_variation_objective_function(imgs, self.eps)


def total_variation_square_op(imgs: TensorLike):
    imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
    imgs4D = tf.reshape(imgs_tf, shape=tf.concat([tf.shape(imgs_tf)[:3], [1]], axis=0))
    diff_x = imgs4D[:, :, 1:] - imgs4D[:, :, :-1]
    diff_y = imgs4D[:, 1:] - imgs4D[:, :-1]
    total_variation = (tf.square(diff_x[:, :-1]) + tf.square(diff_y[:, :, :-1]))
    return total_variation


def TV_square_diff_op(imgs1: TensorLike, imgs2: TensorLike):
    return total_variation_square_op(imgs1 - imgs2)


def TV_square_diff_norm(imgs1: TensorLike, imgs2: TensorLike):
    imgs1_tf = tf.convert_to_tensor(imgs1)
    imgs2_tf = tf.convert_to_tensor(imgs2)
    return tf.reduce_mean(TV_square_diff_op(imgs1_tf, imgs2_tf))


def ssims_tf(imgs1, imgs2):
    """Result is of shape (N,) or ()."""
    imgs1_tf = tf.convert_to_tensor(imgs1)
    imgs2_tf = tf.convert_to_tensor(imgs2)
    # tf.image.ssim returns batch of ssims
    ssim_values = tf.image.ssim(imgs1_tf, imgs2_tf, max_val=5000.0 / parameters.HU_TO_CT_SCALING)
    return ssim_values


def shape_to_3D(imgs):
    """Removes squeezable channel dimension, if there is one.
    Args:
        imgs: tensor or numpy array in custom format.
    Returns:
        If last dimension is squeezable, i.e. C=1, then it is removed.
    """
    imgs_tf = tf.convert_to_tensor(imgs)
    if tf.shape(imgs_tf)[-1] == 1:
        imgs_tf = tf.squeeze(imgs_tf, axis=[-1])
    return imgs_tf


def mean_absolute_errors_tf(imgs1, imgs2):
    """Any of the (N)HW(C) format work for C=1. Result is of shape (N,) or ()."""
    imgs1_tf = shape_to_3D(imgs1)
    imgs2_tf = shape_to_3D(imgs2)
    return tf.reduce_mean(tf.abs(imgs1_tf-imgs2_tf), axis=[-2, -1])


def relative_errors_tf(y_true, y_pred):
    """Any of the (N)HW(C) format work for C=1. Result is of shape (N,) or ()."""
    return tf.norm(shape_to_3D(y_true - y_pred), axis=[-2, -1]) / tf.norm(shape_to_3D(y_true), axis=[-2, -1])


def mean_squares_tf(imgs):
    """Result is of shape (N,) or ()."""
    imgs_tf = shape_to_3D(imgs)
    return tf.reduce_mean(tf.square(imgs_tf), axis=[-2, -1])
