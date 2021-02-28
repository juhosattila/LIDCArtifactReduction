import numpy as np
import tensorflow as tf

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform
from LIDCArtifactReduction.tf_image import scale_Gray2Radio


class DicomOfflineTransformation:
    def __call__(self, data_batch, intercepts, slopes):
        pass


class DummyOfflineTransformation(DicomOfflineTransformation):
    def __call__(self, data_batch, intercepts, slopes):
        return list(zip(data_batch, data_batch))


class ResizeRescaleRadonOfflineTransformation(DicomOfflineTransformation):
    """Resizes images to the input size of the Radon transformation"""
    def __init__(self, radon_geometry: RadonGeometry, radon_transformation: ForwardprojectionRadonTransform):
        self._resize_target = [radon_geometry.volume_img_width, radon_geometry.volume_img_width]
        self._radon_transformation = radon_transformation

    def __call__(self, data_batch, intercepts, slopes):
        scaled_data_batch, data_sino_batch = self._transformation(data_batch, intercepts, slopes)
        return list(zip(scaled_data_batch, data_sino_batch))

    def _transformation(self, data_batch, intercepts, slopes):
        data_batch_tf = tf.convert_to_tensor(data_batch, dtype=tf.float32)
        intercepts_tf = tf.convert_to_tensor(intercepts, dtype=tf.float32)
        slopes_tf = tf.convert_to_tensor(slopes, dtype=tf.float32)
        scaled_data, data_sino = self._tf_transformation(data_batch_tf, intercepts_tf, slopes_tf)
        return np.array(scaled_data, dtype=np.float32), np.array(data_sino, dtype=np.float32)

    # Always convert arguments to tensor before passing and make sure, that shape is constant,
    # otherwise new and new graphs will be created, slowing down the execution.
    # In this case size of tensor is varying, hence it is slow. Do not use tf.function.
    # Toggle for performance, if needed and sizes are fixed.
    # @tf.function
    def _tf_transformation(self, data_batch, intercepts, slopes):
        data_batch = tf.expand_dims(data_batch, axis=-1)
        resized_data = tf.image.resize(data_batch, size=self._resize_target)
        scaled_data = scale_Gray2Radio(resized_data, intercepts, slopes)
        data_sino = self._radon_transformation.forwardproject(scaled_data)
        return scaled_data, data_sino
