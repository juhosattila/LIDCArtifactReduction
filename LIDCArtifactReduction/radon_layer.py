import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer

import LIDCArtifactReduction
from LIDCArtifactReduction.radon_transformation import ParallelRadonTransform
from LIDCArtifactReduction.radon_params import RadonParams


@tf.keras.utils.register_keras_serializable(LIDCArtifactReduction.__name__)
class RadonLayer(Layer):
    def __init__(self, angles_or_params: np.ndarray or RadonParams,
                 is_degree=True, projection_width: int = None,
                 name=None):
        """
        Args:
            projection_width: is the width of each projection, given in pixels. So it is an integer.
        """
        super().__init__(trainable=False, name=name)

        self._radon_params = angles_or_params
        if not isinstance(angles_or_params, RadonParams):
            self._radon_params = RadonParams(angles=angles_or_params, is_degree=is_degree,
                                             projection_width=projection_width)

        # To be assigned after call to `build'.
        self._radon_transformation = None
        self._img_side_length = None

    def build(self, input_shape):
        """We assume that H=W and C=1, if there is a channel dimension."""
        if input_shape[1] != input_shape[2] or \
                np.size(input_shape) == 3 and input_shape[3] != 1:
            raise ValueError("Input shape is assumed H=W and C=1, if there is a channel dimension\n"
                             "Class ParallelRadonLayer works with squared images.\n"
                             f"Error occured in layer {self.name}. Received: {input_shape}.")

        super().build(input_shape)
        self._img_side_length = input_shape[1]
        self._set_projection_width()
        self._radon_transformation = ParallelRadonTransform(self._img_side_length, self._radon_params)

    def _set_projection_width(self):
        if self._radon_params.projection_width is None:
            self._radon_params.projection_width = self._img_side_length

    def get_config(self):
        config = super().get_config()
        config.update({'params': self._radon_params.toJson()})
        return config

    # TODO: not tested, if it works
    @classmethod
    def from_config(cls, config):
        radon_params = RadonParams.fromJson(config['params'])
        print("Suddenly from_config works.")
        return cls(angles_or_params=radon_params, name=config['name'])

    def call(self, inputs, **kwargs):
        return self._radon_transformation.apply(inputs)
