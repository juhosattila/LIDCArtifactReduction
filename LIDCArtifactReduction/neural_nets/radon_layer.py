import tensorflow as tf
from tensorflow.keras.layers import Layer

import LIDCArtifactReduction
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform, \
    ARTRadonTransform


@tf.keras.utils.register_keras_serializable(LIDCArtifactReduction.__name__)
class ForwardRadonLayer(Layer):
    def __init__(self, radon_transformation: ForwardprojectionRadonTransform, name=None):
        super().__init__(trainable=False, name=name)
        self._radon_transformation = radon_transformation

    def call(self, inputs, **kwargs):
        return self._radon_transformation.forwardproject(inputs)


    # # Sample for serialization.
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({'params': self._radon_transformation.toJson()})
    #     return config

    def get_config(self):
        raise NotImplementedError()

    # # Sample for deserialisation. To be tested.
    # @classmethod
    # def from_config(cls, config):
    #     radon_params = RadonGeometry.fromJson(config['params'])
    #     print("Suddenly from_config works.")
    #     return cls(angles_or_params=radon_params, name=config['name'])
    #


@tf.keras.utils.register_keras_serializable(LIDCArtifactReduction.__name__)
class ARTRadonLayer(Layer):
    def __init__(self, radon_transformation: ARTRadonTransform, name=None):
        super().__init__(trainable=False, name=name)
        self._radon_transformation = radon_transformation

    def call(self, inputs, **kwargs):
        volumes, sinos = inputs
        return self._radon_transformation.ART_step(volumes, sinos)

    def get_config(self):
        raise NotImplementedError()
