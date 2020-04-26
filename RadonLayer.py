import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tf_image import ParallelRadonTransform


class RadonParams:
    def __init__(self, angles: np.ndarray, is_degree=True, projection_width: int = None):
        self.angles = angles
        self.is_degree = is_degree
        self.projection_width = projection_width


class RadonLayer(Layer):
    def __init__(self, angles_or_params: np.ndarray or RadonParams,
                 is_degree=True, projection_width: int = None,
                 name=None):
        """
        Args:
            projection_width: is the width of each projection, given in pixels. So it is an integer.
        """
        super().__init__(trainable=False, name=name)

        if isinstance(angles_or_params, RadonParams):
            self._angles = angles_or_params.angles
            self._is_degree = angles_or_params.is_degree
            self._projection_width = angles_or_params.projection_width
        else:
            self._angles = angles_or_params
            self._is_degree = is_degree
            self._projection_width = projection_width

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
        self._projection_width = self._projection_width \
            if self._projection_width is not None else self._img_side_length
        self._radon_transformation = ParallelRadonTransform(self._img_side_length, self._angles, self._is_degree,
                                                            self._projection_width)

    @tf.function
    def call(self, inputs, **kwargs):
        return self._radon_transformation.apply(inputs)

    # TODO:
    def get_config(self):
        raise NotImplementedError(f"get_config for {self.__class__.name} was not implemented.")
        # config = super().get_config()
        # config.update({'theta': self.theta})
        # return config
