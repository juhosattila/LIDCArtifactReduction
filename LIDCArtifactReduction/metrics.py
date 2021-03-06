from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.metrics import Metric, Mean, RootMeanSquaredError, MeanAbsoluteError, MeanSquaredError

from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.image import tf_image
from LIDCArtifactReduction.image.tf_image import scale_Radiodiff2HUdiff, ssims_tf, mean_squares_tf, signal2noise_tf, \
    reference2noise_tf
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform


class MeanBasedMetric(Metric):
    def __init__(self, name, dtype):
        super().__init__(name, dtype=dtype)
        self._mean = Mean(dtype=dtype)

    @abstractmethod
    def _objective_function(self, y_true, y_pred):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self._objective_function(y_true, y_pred)
        self._mean.update_state(values=values, sample_weight=sample_weight)

    def result(self):
        return self._mean.result()

    def reset_states(self):
        self._mean.reset_states()


class StandardVarianceBasedMetric(Metric):
    def __init__(self, name, dtype):
        super().__init__(name, dtype=dtype)
        self._mean = Mean(dtype=dtype)
        self._square_mean = Mean(dtype=dtype)

    @abstractmethod
    def _objective_function(self, y_true, y_pred):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self._objective_function(y_true, y_pred)
        self._mean.update_state(values=values, sample_weight=sample_weight)
        self._square_mean.update_state(values=tf.square(values), sample_weight=sample_weight)

    def result(self):
        return tf.sqrt(self._square_mean.result() - tf.square(self._mean.result()))

    def reset_states(self):
        self._mean.reset_states()
        self._square_mean.reset_states()


class SumSquaredError(MeanBasedMetric):
    """This metric gives the mean of summed squares."""
    def __init__(self, name='sum_square_error', dtype=None):
        super().__init__(name, dtype=dtype)

    def _objective_function(self, y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true-y_pred), axis=range(1, tf.rank(y_true)))


class HU_RMSE(RootMeanSquaredError):
    def __init__(self, name='HU_RMSE', dtype=None):
        super().__init__(name, dtype=dtype)

    def result(self):
        return scale_Radiodiff2HUdiff(super().result())


class HU_MAE(MeanAbsoluteError):
    def __init__(self, name='HU_MAE', dtype=None):
        super().__init__(name=name, dtype=dtype)

    def result(self):
        return scale_Radiodiff2HUdiff(super().result())


class ReconstructionReference2Noise(MeanBasedMetric):
    def __init__(self, name='rec_ref2noise', dtype=None):
        super().__init__(name, dtype=dtype)

    def _objective_function(self, y_true, y_pred):
        ref = 1000.0 / parameters.HU_TO_CT_SCALING
        return reference2noise_tf(y_true, y_pred, ref=ref)


class Signal2Noise(MeanBasedMetric):
    def __init__(self, name='signal2noise', dtype=None):
        super().__init__(name, dtype=dtype)

    def _objective_function(self, y_true, y_pred):
        return signal2noise_tf(y_true, y_pred)


class Signal2NoiseStandardDeviance(StandardVarianceBasedMetric):
    def __init__(self, name='signa2noise_std', dtype=None):
        super().__init__(name, dtype=dtype)

    def _objective_function(self, y_true, y_pred):
        return signal2noise_tf(y_true, y_pred)


class SSIM(MeanBasedMetric):
    def __init__(self, name='SSIM', dtype=None):
        super().__init__(name, dtype=dtype)

    def _objective_function(self, y_true, y_pred):
        return ssims_tf(y_pred, y_true)


class MeanSquare(Mean):
    def __init__(self, name='mean_square', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, values, sample_weight=None):
        update_values = mean_squares_tf(values)
        return super().update_state(values=update_values, sample_weight=sample_weight)


class MeanSumSquare(Mean):
    def __init__(self, name='mean_sum_square', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, imgs, sample_weight=None):
        values = tf.reduce_sum(tf.square(imgs), axis=range(1, tf.rank(imgs)))
        super().update_state(values=values, sample_weight=sample_weight)


class RelativeError(MeanBasedMetric):
    def __init__(self, name='rel_error', dtype=None):
        super().__init__(name, dtype=dtype)

    def _objective_function(self, y_true, y_pred):
        return tf_image.relative_errors_tf(y_true, y_pred)


class RadonMeanSquaredError(MeanSquaredError):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform, name='radon_mse', dtype=None):
        super().__init__(name, dtype=dtype)
        self._radon_transformation = radon_transformation

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(
            self._radon_transformation.forwardproject(y_true),
            self._radon_transformation.forwardproject(y_pred),
            sample_weight=sample_weight)


class RadonRelativeError(RelativeError):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform, name='radon_rel_error', dtype=None):
        super().__init__(name, dtype=dtype)
        self._radon_transformation = radon_transformation

    def _objective_function(self, y_true, y_pred):
        return super()._objective_function(
            self._radon_transformation.forwardproject(y_true),
            self._radon_transformation.forwardproject(y_pred))
