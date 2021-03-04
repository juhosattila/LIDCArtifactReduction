from typing import List
from abc import abstractmethod
import numbers

from tensorflow.keras.layers import Input, Conv2D, \
    BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose
from tensorflow.keras import regularizers

from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface


class ResidualUNetAbstract(ModelInterface):
    def __init__(self, volume_img_width: int, has_batch_norm=True, has_dropout=False,
                 has_activation_after_upsampling=False, conv_regularizer=None,
                 name=None, weight_dir=None, input_name=None, output_name=None,
                 output_difference_layer: bool = False, difference_name=None):
        """
        Args:
            conv_regularizer: either a  regularizer class or a number for weight of l2 reg
        """
        super().__init__(name=name, weight_dir=weight_dir)

        # Should be tuple of integers that are divisible by 2^4
        self._input_shape = (volume_img_width, volume_img_width)

        self._has_batch_norm = has_batch_norm
        self._batch_norm_layers: List[BatchNormalization] = []
        self._has_dropout = has_dropout
        self._dropout_layers: List[Dropout] = []
        self._has_activation_after_upsampling = has_activation_after_upsampling
        self._conv_layer_initalizer = 'he_normal'

        if conv_regularizer is None:
            self._conv_layer_regualizer = None
        elif isinstance(conv_regularizer, numbers.Number):
            self._conv_layer_regualizer = regularizers.l2(conv_regularizer)
        else:
            self._conv_layer_regualizer = conv_regularizer

        self._input_name = input_name or 'input_layer'
        self._output_name = output_name or 'output_layer'
        self._difference_name = difference_name or 'difference_layer'
        self._output_difference_layer = output_difference_layer

        self._model = None
        self._input_layer = None
        self._output_layer = None
        self._difference_layer = None

        self._build_model()

    def _input(self):
        return Input(self._input_shape + (1,), name=self._input_name)

    def _conv_k3_activation(self, filters: int):
        return Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                      activation='relu', kernel_initializer=self._conv_layer_initalizer,
                      kernel_regularizer=self._conv_layer_regualizer)

    def _possible_batch_norm(self):
        if self._has_batch_norm:
            batch_norm_layer = BatchNormalization()
            self._batch_norm_layers.append(batch_norm_layer)
            return lambda x: batch_norm_layer(x)
        return lambda x: x

    def _conv_k3_activation_possible_batchnorm(self, filters: int):
        conv = self._conv_k3_activation(filters)
        possible_batch_norm = self._possible_batch_norm()
        return lambda x: possible_batch_norm(conv(x))

    def _max_pooling(self):
        return MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    def _possible_dropout(self):
        func = lambda x: x
        if self._has_dropout:
            dropout_layer = Dropout(0.5)
            func = lambda x: dropout_layer(x)
            self._dropout_layers.append(dropout_layer)
        return func

    def _resize_conv(self, filters: int):
        # TODO: in upsampling what interpolation technique to use
        # Difference between the performances of transposed convolution and upsampling + conv are discussed here:
        #   https://distill.pub/2016/deconv-checkerboard/
        upsampling = UpSampling2D(size=(2, 2), interpolation='nearest')
        activation = 'relu' if self._has_activation_after_upsampling else 'linear'
        conv = Conv2D(filters=filters, kernel_size=(2, 2), padding='same',
                      activation=activation, kernel_initializer=self._conv_layer_initalizer,
                      kernel_regularizer=self._conv_layer_regualizer)
        func = lambda x: conv(upsampling(x))
        return func

    def _trans_conv(self, filters: int):
        activation = 'relu' if self._has_activation_after_upsampling else 'linear'
        return Conv2DTranspose(filters, kernel_size=(2,2), padding='same', strides=(2,2),
                        activation=activation, kernel_initializer=self._conv_layer_initalizer,
                        kernel_regularizer=self._conv_layer_regualizer)

    def get_difference_layer(self):
        return Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                      kernel_initializer=self._conv_layer_initalizer,
                      kernel_regularizer=self._conv_layer_regualizer,
                      name=self._difference_name)

    @abstractmethod
    def _build_model(self):
        pass

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def output_layer(self):
        return self._output_layer

    @property
    def difference_layer(self):
        """Returns output_layer - input_layer."""
        return self._difference_layer
