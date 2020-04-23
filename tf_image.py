import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from typing import List

from tensorflow_addons.utils.types import TensorLike


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
    transformation_parameters = _get_rotate_translate_resampling_params(input_img_side_length,
                                                                        _angle, translate, real_output_img_side_length)
    return tfa.image.transform(images=input_img,
                               transforms=transformation_parameters,
                               interpolation='BILINEAR',
                               output_shape=[real_output_img_side_length, real_output_img_side_length])


def _get_rotate_translate_resampling_params(input_img_side_length, angle,
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


class ParallelRadonTransform:
    """Class able to compute the Parallel Radon Transform for a batch of images.

    Only squared images are supported for the time being.

    Example:
        radon_trafo = RadonTransform(512, np.linspace(0., 180., 180) * math.pi / 180)
        sinogram = radon_trafo.apply(imgs)
    """

    def __init__(self, img_side_length: int, angles, is_degree: bool = True, projection_width: int = None):
        """
        Args:
            img_side_length: the width of an input image. Remember that input images should be squared.
            projection_width: should be understood in number of pixels
        """
        self._input_shape_with_channel = tf.convert_to_tensor([img_side_length, img_side_length, 1])
        self._projection_width = tf.convert_to_tensor(projection_width) \
            if projection_width is not None else img_side_length
        self._angles = tf.constant(angles if not is_degree else np.deg2rad(angles), dtype=tf.float32)

        self._transformation_params = tf.map_fn(lambda angle:
                _get_rotate_translate_resampling_params(img_side_length, angle,
                                                        output_img_side_length=self._projection_width),
                self._angles
            )

    def __call__(self, img_or_imgs: List[TensorLike]):
        return self.apply(img_or_imgs)

    def apply(self, imgs: List[TensorLike]):
        """
         Args:
            imgs: List of Tensors in NHW or NHWC mode. Remember that H=W. C should be 1

        Raises:
            InvalidArgumentError: if H =/= W
        """
        imgs4D = tf.reshape(imgs, shape=tf.concat([[tf.shape(imgs)[0]], self._input_shape_with_channel], axis=0))

        apply_one_projection = lambda params: tf.reduce_sum \
            (
                tfa.image.transform(images=imgs4D, transforms=params,
                                    interpolation='BILINEAR',
                                    output_shape=[self._projection_width, self._projection_width]),
                axis=1
            )

        radon_bad_transpose = tf.map_fn(apply_one_projection, self._transformation_params)
        return tf.transpose(
            radon_bad_transpose,
            perm=[1, 0, 2, 3]
        )

    def apply_one_image(self, img):
        return self.apply(tf.expand_dims(img, axis=0))
