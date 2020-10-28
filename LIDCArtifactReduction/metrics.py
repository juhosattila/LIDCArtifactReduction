import tensorflow as tf
from tensorflow.keras.metrics import Metric, Mean, RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError

from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.tf_image import scale_Radio2HU, ssim_tf, mean_absolute_error_tf, shape_to_3D


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
        return  multiplier * tf.math.log( tf.square(1000.0 / parameters.HU_TO_CT_SCALING) / (super().result()) )


class SSIM(Metric):
    def __init__(self, name='SSIM', dtype=None):
        super().__init__(name, dtype=dtype)
        self._mean = Mean(dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim_value = ssim_tf(y_pred, y_true)
        return self._mean.update_state(ssim_value, sample_weight=sample_weight)

    def result(self):
        return self._mean.result()

    def reset_states(self):
        self._mean.reset_states()


class MeanSquare(Mean):
    def __init__(self, name='mean_square', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, values, sample_weight=None):
        # TODO: find out the dimension of gradients and specify axis.
        # Make it work for different batchsizes in NHWC.
        # Otherwise its ok because batch-es have fixed size.
        update_values = tf.reduce_mean(tf.square(values))#, axis=[])
        return super().update_state(values=update_values, sample_weight=sample_weight)


class RelativeError(Metric):
    def __init__(self, name='rel_error', dtype=None):
        super().__init__(name, dtype=dtype)
        self._mean = Mean(dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        rel_error = tf.norm(shape_to_3D(y_true - y_pred), axis=[-2, -1]) / tf.norm(shape_to_3D(y_true), axis=[-2, -1])
        self._mean.update_state(values=rel_error)

    def result(self):
        return self._mean.result()

    def reset_states(self):
        self._mean.reset_states()


# TODO: refactor. Problem with dimensions. I do not know yet.
def ssim_np(img1, img2):
    return ssim_tf(img1, img2).numpy()


def mean_absolute_errors_np(imgs1, imgs2):
    return mean_absolute_error_tf(imgs1, imgs2).numpy()


def mean_absolute_error_HU_np(imgs1, imgs2):
    return mean_absolute_error_tf(
                scale_Radio2HU(imgs1),
                scale_Radio2HU(imgs2)).numpy()
