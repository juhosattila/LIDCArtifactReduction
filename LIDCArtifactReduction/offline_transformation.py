import tensorflow as tf

from LIDCArtifactReduction.tf_image import scale_HU2Radio
from LIDCArtifactReduction.radon_transformation import ParallelRadonTransform
from LIDCArtifactReduction.radon_params import RadonParams


class DicomOfflineTransformation:
    def __call__(self, data_batch, intercepts, slopes):
        pass


class DummyOfflineTransformation(DicomOfflineTransformation):
    def __call__(self, data_batch, intercepts, slopes):
        return list(zip(data_batch, data_batch))


class ResizeRescaleRadonOfflineTransformation(DicomOfflineTransformation):
    """Resizes images to the target size of the received Radon transformation"""
    def __init__(self, resize_size: int, radon_params: RadonParams):
        self._resize_target = [resize_size, resize_size]
        self._radon_transformation = ParallelRadonTransform(img_side_length=resize_size,
                                                            angles_or_params=radon_params)

    def __call__(self, data_batch, intercepts, slopes):
        scaled_data_batch, data_sino_batch = self._tf_transformation(data_batch, intercepts, slopes)
        return list(zip(scaled_data_batch, data_sino_batch))

    # TODO : comment out directive
    # @tf.function
    def _tf_transformation(self, data_batch, intercepts, slopes):
        data_tf = tf.convert_to_tensor(data_batch, dtype=tf.float32)
        data_tf = tf.expand_dims(data_tf, axis=-1)
        resized_data = tf.image.resize(data_tf, size=self._resize_target)
        scaled_data = scale_HU2Radio(resized_data, intercepts, slopes)
        data_sino = self._radon_transformation(scaled_data)
        return scaled_data.numpy(), data_sino.numpy()

