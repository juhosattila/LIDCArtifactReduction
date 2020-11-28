import tensorflow as tf
from tensorflow.keras.metrics import Metric, Mean, RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError

from LIDCArtifactReduction import parameters, tf_image
from LIDCArtifactReduction.tf_image import scale_Radio2HU, ssims_tf, mean_squares_tf


class HU_RMSE(RootMeanSquaredError):
    def __init__(self, name='HU_RMSE', dtype=None):
        super().__init__(name, dtype=dtype)

    def result(self):
        return scale_Radio2HU(super().result())


class HU_MAE(MeanAbsoluteError):
    def __init__(self, name='HU_MAE', dtype=None):
        super().__init__(name, dtype=dtype)

    def result(self):
        return scale_Radio2HU(super().result())


class RadioSNR(MeanSquaredError):
    def __init__(self, name='RadioSNR', dtype=None):
        super().__init__(name, dtype=dtype)

    def result(self):
        # to make it a usual SNR definition:
        multiplier = 10.0 / tf.math.log(10.0)  # = 4.3429
        return multiplier * tf.math.log( tf.square(1000.0 / parameters.HU_TO_CT_SCALING) / (super().result()) )


class SSIM(Metric):
    def __init__(self, name='SSIM', dtype=None):
        super().__init__(name, dtype=dtype)
        self._mean = Mean(dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim_value = ssims_tf(y_pred, y_true)
        return self._mean.update_state(ssim_value, sample_weight=sample_weight)

    def result(self):
        return self._mean.result()

    def reset_states(self):
        self._mean.reset_states()


class MeanSquare(Mean):
    def __init__(self, name='mean_square', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, values, sample_weight=None):
        update_values = mean_squares_tf(values)
        return super().update_state(values=update_values, sample_weight=sample_weight)


class RelativeError(Metric):
    def __init__(self, name='rel_error', dtype=None):
        super().__init__(name, dtype=dtype)
        self._mean = Mean(dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        rel_error = tf_image.relative_errors_tf(y_true, y_pred)
        self._mean.update_state(values=rel_error)

    def result(self):
        return self._mean.result()

    def reset_states(self):
        self._mean.reset_states()
