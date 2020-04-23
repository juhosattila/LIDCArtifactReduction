import os
from typing import List
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, \
    BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Add

from RadonLayer import RadonLayer, RadonParams


class ModelInterface:
    object_counter = {}

    def __init__(self, name):
        valid_name = name
        if valid_name is None:
            class_name = type(self).__name__
            ModelInterface.object_counter[class_name] = ModelInterface.object_counter.get(class_name, 0) + 1
            valid_name = class_name + '_' + str(ModelInterface.object_counter[class_name])
        self._name = valid_name
        self._model = None

    @property
    def name(self):
        return self._name

    def summary(self):
        self._model.summary()

    def plot_model(self, to_file=None, show_shapes=True):
        dir = 'model_plots'
        if not os.path.exists(dir):
            os.mkdir(dir)
        valid_to_file = to_file if to_file is not None else (self.name + '.png')
        final_to_file = os.path.join(dir, valid_to_file)
        plot_model(self._model, final_to_file, show_shapes)


# TODO: decide whether dropout and batchnorm are necessary
# TODO: in upsampling what interpolation technique to use
# TODO: setting training needs compiling
class DCAR_UNet(ModelInterface):
    """Defaults are made according to article
    Huang, Wurfle: Some investigations on Robustness of Deep Learning in Limited Andle Tomography (2018).
    This means that by default there is natch norm, there is NO dropout and activation after upsampling.

    Args:
        input_shape: tuple of integers that are divisible by 2^4
    """

    def __init__(self, input_shape=(256, 256), has_batch_norm=True, has_dropout=False,
                 has_activation_after_upsampling=False, conv_width_factor=64, name=None):
        super().__init__(name)
        self._input_shape = input_shape
        self._has_batch_norm = has_batch_norm
        self._batch_norm_layers: List[BatchNormalization] = []
        self._has_dropout = has_dropout
        self._dropout_layers: List[Dropout] = []
        self._has_activation_after_upsampling = has_activation_after_upsampling
        self._conv_layer_initalizer = 'he_normal'
        self._conv_width_factor = conv_width_factor

        self._model, self._input_layer, self._output_layer = self._build_model()

    def _set_training(self, training: bool):
        for batch_norm_layer in self._batch_norm_layers:
            batch_norm_layer.training = training
        for dropout_layer in self._dropout_layers:
            dropout_layer.training = training

    def _conv_activation_possible_batchnorm(self, filter_factor: int):
        conv = Conv2D(filters=filter_factor * self._conv_width_factor, kernel_size=(3, 3), padding='same',
                      activation='relu', kernel_initializer='he_uniform')
        func = lambda x: conv(x)
        if self._has_batch_norm:
            batch_norm_layer = BatchNormalization()
            func = lambda x: batch_norm_layer(conv(x))
            self._batch_norm_layers.append(batch_norm_layer)
        return func

    def _max_pooling(self):
        return MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    def _possible_dropout(self):
        func = lambda x: x
        if self._has_dropout:
            dropout_layer = Dropout(0.5)
            func = lambda x: dropout_layer(x)
            self._dropout_layers.append(dropout_layer)
        return func

    def _resize_conv(self, filter_factor: int):
        upsampling = UpSampling2D(size=(2, 2), interpolation='nearest')
        activation = 'relu' if self._has_activation_after_upsampling else 'linear'
        conv = Conv2D(filters=filter_factor * self._conv_width_factor, kernel_size=(2, 2), padding='same',
                      activation=activation, kernel_initializer=self._conv_layer_initalizer)
        func = lambda x: conv(upsampling(x))
        return func

    def _build_model(self):
        input_layer = Input(self._input_shape + (1,))

        d0 = input_layer
        d0 = self._conv_activation_possible_batchnorm(1)(d0)
        d0 = self._conv_activation_possible_batchnorm(1)(d0)

        d1 = self._max_pooling()(d0)
        d1 = self._conv_activation_possible_batchnorm(2)(d1)
        d1 = self._conv_activation_possible_batchnorm(2)(d1)

        d2 = self._max_pooling()(d1)
        d2 = self._conv_activation_possible_batchnorm(4)(d2)
        d2 = self._conv_activation_possible_batchnorm(4)(d2)

        d3 = self._max_pooling()(d2)
        d3 = self._conv_activation_possible_batchnorm(8)(d3)
        d3 = self._conv_activation_possible_batchnorm(8)(d3)

        d3 = self._possible_dropout()(d3)

        d4 = self._max_pooling()(d3)
        d4 = self._conv_activation_possible_batchnorm(16)(d4)
        d4 = self._conv_activation_possible_batchnorm(16)(d4)

        d4 = self._possible_dropout()(d4)

        u3 = self._resize_conv(8)(d4)
        u3 = Concatenate()([d3, u3])
        u3 = self._conv_activation_possible_batchnorm(8)(u3)
        u3 = self._conv_activation_possible_batchnorm(8)(u3)

        u2 = self._resize_conv(4)(u3)
        u2 = Concatenate()([d2, u2])
        u2 = self._conv_activation_possible_batchnorm(4)(u2)
        u2 = self._conv_activation_possible_batchnorm(4)(u2)

        u1 = self._resize_conv(2)(u2)
        u1 = Concatenate()([d1, u1])
        u1 = self._conv_activation_possible_batchnorm(2)(u1)
        u1 = self._conv_activation_possible_batchnorm(2)(u1)

        u0 = self._resize_conv(1)(u1)
        u0 = Concatenate()([d0, u0])
        u0 = self._conv_activation_possible_batchnorm(1)(u0)
        u0 = self._conv_activation_possible_batchnorm(1)(u0)

        diff_layer = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                            kernel_initializer=self._conv_layer_initalizer)(u0)

        output_layer = Add()([input_layer, diff_layer])

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        return model, input_layer, output_layer

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def output_layer(self):
        return self._output_layer


class DCAR_TrainingNetwork(ModelInterface):
    def __init__(self, radon_params: RadonParams, target_model: DCAR_UNet = None, name=None):
        super().__init__(name)
        self._target_model = target_model if target_model is not None else DCAR_UNet()
        self._input_shape = target_model.input_shape

        self._radon_params = radon_params

        self._model = None
        self._input_layer = None
        self._expected_output_layer = None
        self._expected_Radon_layer = None

        self._build_model()

    def _build_model(self):
        target_input_layer = self._target_model.input_layer
        target_output_layer = self._target_model.output_layer

        expected_output_layer = target_output_layer
        expected_Radon_layer = RadonLayer(**self._radon_params.__dict__)(expected_output_layer)

        model = Model(inputs=target_input_layer, outputs=[expected_output_layer, expected_Radon_layer],
                      name=self.name)

        self._input_layer = target_input_layer
        self._expected_output_layer = expected_output_layer
        self._expected_Radon_layer = expected_Radon_layer
        self._model = model

    @property
    def target_model(self):
        return self._target_model
