import tensorflow as tf
from tensorflow.keras.layers import Layer

import LIDCArtifactReduction
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform, \
    RadonTransform


@tf.keras.utils.register_keras_serializable(LIDCArtifactReduction.__name__)
class ForwardRadonLayer(Layer):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform, name=None):
        super().__init__(trainable=False, name=name)
        self._radon_transformation = radon_transformation

    def call(self, inputs, **kwargs):
        return self._radon_transformation.forwardproject(inputs)

    # def build(self, input_shape):
    #     """We assume that H=W and C=1, if there is a channel dimension."""
    #     if input_shape[1] != input_shape[2] or \
    #             np.size(input_shape) == 3 and input_shape[3] != 1:
    #         raise ValueError("Input shape is assumed H=W and C=1, if there is a channel dimension\n"
    #                          "Class ParallelRadonLayer works with squared images.\n"
    #                          f"Error occured in layer {self.name}. Received: {input_shape}.")
    #
    #     super().build(input_shape)
    #     self._img_side_length = input_shape[1]
    #     self._set_projection_width()
    #     self._radon_transformation = ForwardprojectionParallelRadonTransform(self._img_side_length, self._radon_geometry)

    # def _set_projection_width(self):
    #     if self._radon_params.projection_width is None:
    #         self._radon_params.projection_width = self._img_side_length

    # TODO: if serialization is not necessary, just delete
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({'params': self._radon_transformation.toJson()})
    #     return config

    def get_config(self):
        raise NotImplementedError()

    #
    # # TODO: not tested, if it works
    # @classmethod
    # def from_config(cls, config):
    #     radon_params = RadonGeometry.fromJson(config['params'])
    #     print("Suddenly from_config works.")
    #     return cls(angles_or_params=radon_params, name=config['name'])
    #


@tf.keras.utils.register_keras_serializable(LIDCArtifactReduction.__name__)
class ARTRadonLayer(Layer):
    def __init__(self, radon_transformation: RadonTransform, name=None):
        super().__init__(trainable=False, name=name)
        self._radon_transformation = radon_transformation

    def call(self, inputs, **kwargs):
        volumes, sinos = inputs
        computed_sinos = self._radon_transformation.forwardproject(volumes)


    def get_config(self):
        raise NotImplementedError()
