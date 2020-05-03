import numpy as np
from typing import List

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.utils.types import TensorLike

from tf_image import get_rotate_translate_resampling_params


class RadonParams:
    def __init__(self, angles: np.ndarray, is_degree=True, projection_width: int = None):
        self.angles = angles
        self.is_degree = is_degree
        self.projection_width = projection_width


class ParallelRadonTransform:
    """Class able to compute the Parallel Radon Transform for a batch of images.

    Only squared images are supported for the time being.

    Example:
        radon_trafo = RadonTransform(512, np.linspace(0., 180., 180) * math.pi / 180)
        sinogram = radon_trafo.apply(imgs)
    """

    def __init__(self, img_side_length: int, angles_or_params, is_degree: bool = True, projection_width: int = None):
        """
        Args:
            img_side_length: the width of an input image. Remember that input images should be squared.
            projection_width: should be understood in number of pixels
        """
        angles = angles_or_params
        if isinstance(angles_or_params, RadonParams) is not None:
            angles = angles_or_params.angles
            is_degree = angles_or_params.is_degree
            projection_width = angles_or_params.projection_width

        self._input_shape_with_channel = tf.convert_to_tensor([img_side_length, img_side_length, 1])
        self._projection_width = tf.convert_to_tensor(projection_width) \
            if projection_width is not None else img_side_length
        self._angles = tf.constant(angles if not is_degree else np.deg2rad(angles), dtype=tf.float32)

        self._transformation_params = tf.map_fn(lambda angle:
                get_rotate_translate_resampling_params(img_side_length, angle,
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
