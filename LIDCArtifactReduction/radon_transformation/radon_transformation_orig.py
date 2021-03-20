import numpy as np
from skimage.transform import iradon

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.utils.types import TensorLike

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform, \
    RadonTransform
from LIDCArtifactReduction.image.tf_image import get_rotate_translate_resampling_params


class ForwardprojectionParallelRadonTransform(ForwardprojectionRadonTransform):
    """Class able to compute the Parallel Radon Transform for a batch of images.

    Only squared images are supported for the time being.
    """

    def __init__(self, geometry: RadonGeometry):
        nr_projections = geometry.nr_projections
        projection_width = geometry.projection_width
        volume_img_width = geometry.volume_img_width

        self._input_shape_with_channel = tf.convert_to_tensor([volume_img_width, volume_img_width, 1])
        self._projection_width = tf.convert_to_tensor(projection_width)
        self._angles = tf.constant(np.deg2rad(np.linspace(0., 180., nr_projections)), dtype=tf.float32)

        self._transformation_params = \
            tf.map_fn(lambda angle:
                get_rotate_translate_resampling_params(volume_img_width, angle,
                                                       output_img_side_length=self._projection_width),
                self._angles)

    def forwardproject(self, imgs: TensorLike):

        imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
        imgs4D = tf.reshape(imgs_tf, shape=tf.concat([[tf.shape(imgs_tf)[0]], self._input_shape_with_channel], axis=0))

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
        return self.forwardproject(tf.expand_dims(img, axis=0))


class ParallelRadonTransform(RadonTransform):
    def __init__(self, radon_geometry: RadonGeometry):
        self._geometry = radon_geometry
        self._forward_parallel_radon_transform = ForwardprojectionParallelRadonTransform(radon_geometry)

    def forwardproject(self, imgs: TensorLike):
        return self._forward_parallel_radon_transform.forwardproject(imgs)

    def backproject(self, sinos: TensorLike):
        raise NotImplementedError()

    def invert(self, sinos: TensorLike):
        sinos_np = np.asarray(sinos, dtype=np.float32) # read-only copy is enough, beacuse reshape copies.
        # remove channel dimensions, if any
        sinos_3D = np.reshape(sinos_np, newshape=np.shape(sinos_np)[:3])
        reconstructions = []
        for sino in sinos_3D:
            rec = iradon(np.transpose(sino), theta=np.linspace(0., 180., self._geometry.nr_projections),
                         output_size=self._geometry.volume_img_width, circle=True)
            reconstructions.append(rec)
        # stack and add back channel dimension, even if there wasn't one
        reconstructions = np.expand_dims(np.stack(reconstructions, axis=0), axis=-1)

        return reconstructions.astype(np.float32)
