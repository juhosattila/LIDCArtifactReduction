import abc

import tensorflow as tf
from tensorflow.keras import losses


class IterativeARTRestNetTrainingLoss:
    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass

class RecSinoGradientLoss(IterativeARTRestNetTrainingLoss):
    def __init__(self, reconstructions_output_weight, error_singrom_weight, gradient_weight):
        self._reconstructions_output_weight = tf.convert_to_tensor(reconstructions_output_weight, dtype=tf.float32)
        self._error_sinogram_weight = tf.convert_to_tensor(error_singrom_weight, dtype=tf.float32)
        self._gradient_weight = tf.convert_to_tensor(gradient_weight, dtype=tf.float32)

        self._mse = losses.MeanSquaredError()


    def __call__(self, reconstructions_output, error_sinogram, good_reconstructions, doutput_dinput):
        return self._reconstructions_output_weight * self._mse(reconstructions_output, good_reconstructions) + \
               self._error_sinogram_weight * tf.reduce_mean(tf.square(error_sinogram)) + \
               self._gradient_weight * tf.reduce_mean(tf.square(doutput_dinput))

        # return self._reconstructions_output_weight * tf.reduce_mean(tf.square(reconstructions_output-good_reconstructions)) + \
        #        self._error_sinogram_weight * tf.reduce_mean(tf.square(error_sinogram)) + \
        #        self._gradient_weight * tf.reduce_mean(tf.square(doutput_dinput))
