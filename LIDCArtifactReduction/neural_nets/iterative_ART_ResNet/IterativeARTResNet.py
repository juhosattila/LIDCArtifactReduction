import os

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.radon_layer import ARTRadonLayer
from LIDCArtifactReduction.neural_nets.residual_UNet.residual_UNet_few_batch_norms import ResidualUNetFewBatchNorms
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ARTRadonTransform
from LIDCArtifactReduction.radon_transformation.radon_transformation_pyronn import PyronnParallelARTRadonTransform


class IterativeARTResNet(ModelInterface):

    def __init__(self, radon_geometry: RadonGeometry,
                 radon_transformation: ARTRadonTransform = None,
                 name=None):
        super().__init__(name=name)
        self._radon_geometry = radon_geometry

        # TODO: If adding new options, then refactor and inject dependency.
        self._conv_regularizer = 1e-4
        self._radon_transformation = radon_transformation or PyronnParallelARTRadonTransform(self._radon_geometry)

        self._model = None
        self._imgs_input_layer = None
        self._sinos_input_layer = None
        self._output_layer_or_layers = None
        
        self._build()

    imgs_input_name = 'imgs_input_layer'
    sinos_input_name = 'sinos_input_layer'
    resnet_name = 'resnet_layer'

    def _build(self):
        volume_img_width = self._radon_geometry.volume_img_width
        imgs_input_shape = (volume_img_width, volume_img_width, 1)
        imgs_input_layer = Input(imgs_input_shape, name=IterativeARTResNet.imgs_input_name)

        sinos_input_shape = (self._radon_geometry.nr_projections, self._radon_geometry.projection_width, 1)
        sinos_input_layer = Input(sinos_input_shape, name=IterativeARTResNet.sinos_input_name)

        art_layer = ARTRadonLayer(radon_transformation=self._radon_transformation, name='ART_layer')\
                        ([imgs_input_layer, sinos_input_layer])

        # TODO: Refactor and inject dependency.
        kernel_model = ResidualUNetFewBatchNorms(volume_img_width=self._radon_geometry.volume_img_width,
                                                 conv_regularizer=self._conv_regularizer,
                                                 input_name='kernel_input',
                                                 output_name='kernel_output',
                                                 name=IterativeARTResNet.resnet_name,
                                                 output_difference_layer=True,
                                                 difference_name='kernel_difference')


        output_layer_or_layers = kernel_model(art_layer)
        self._model = Model(inputs=[imgs_input_layer, sinos_input_layer], outputs=output_layer_or_layers)

        self._imgs_input_layer = imgs_input_layer
        self._sinos_input_layer = sinos_input_layer
        self._output_layer_or_layers = output_layer_or_layers


    @property
    def imgs_input_layer(self):
        return self._imgs_input_layer

    @property
    def sinos_input_layer(self):
        return self._sinos_input_layer

    @property
    def output_layer_or_layers(self):
        return self._output_layer_or_layers

    def compile(self):
        raise NotImplementedError()

    def set_training(self, training: bool):
        raise NotImplementedError()
