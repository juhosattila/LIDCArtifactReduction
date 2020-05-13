from tensorflow_addons.utils.types import TensorLike

import tensorflow as tf
import tensorflow_addons as tfa

from LIDCArtifactReduction import parameters


def _get_scale_angle_translate_centre_matrix(scale, angle, translate, centre):
    """Creates a 2D transformation in homogenoues coordinates based on the formula:
    y=S(R(x-c0) + t + c0)

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
    """Scale in a min-max fashion a list of Tensors.

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

    rescaled_tf = imgs_tf * slopes_tf + intercepts_tf + bias_tf

    # scaling of (min,1000) to (0,1)
    # TODO: do it based on minimum, and not 0
    zeroed_out_tf = tf.where(rescaled_tf >= 0, rescaled_tf, tf.zeros_like(rescaled_tf))

    return scale_HU2Radio(zeroed_out_tf)


def total_variation_op(imgs: TensorLike):
    imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
    imgs4D = tf.reshape(imgs_tf, shape=tf.concat([tf.shape(imgs_tf)[:3], [1]], axis=0))
    diff_x = imgs4D[:, :, 1:] - imgs4D[:, :, :-1]
    diff_y = imgs4D[:, 1:] - imgs4D[:, :-1]
    total_variation = tf.sqrt(tf.square(diff_x[:, :-1]) + tf.square(diff_y[:, :, :-1]))
    return total_variation


def total_variation_norm(imgs: TensorLike):
    """
    :param imgs: Should be a 3D NHW or 4D NHWC tensor with C=1.
    """
    total_variation = total_variation_op(imgs)
    tv_norm = tf.reduce_sum(total_variation, axis=[1, 2, 3])

    # # TODO: delete
    # return tv_norm, total_variation

    return tv_norm


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

    # # TODO: delete
    # return reweighted_tv_norm, reweighted_tv

    return reweighted_tv_norm


def sparsity_sum_operator(imgs: TensorLike, eps: float):
    return tf.reduce_sum(tf.math.log(imgs + eps))


def sparsity_mean_operator(imgs: TensorLike, eps: float):
    return tf.reduce_mean(tf.math.log(imgs + eps))


def sparse_total_variation_objective_function(imgs: TensorLike, eps: float):
    total_variation = total_variation_op(imgs)
    return sparsity_mean_operator(total_variation, eps)


class SparseTotalVariationObjectiveFunction:
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, imgs):
        return sparse_total_variation_objective_function(imgs, self.eps)
