from LIDCArtifactReduction.neural_nets.DCAR_TargetAbstract import DCAR_TargetAbstract

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, \
    Concatenate, Add


class DCAR_UNet_ManyBatchNorms(DCAR_TargetAbstract):
    """Defaults are made according to article
    Huang, Wurfle: Some investigations on Robustness of Deep Learning in Limited Andle Tomography (2018).
    This means that by default there is natch norm, there is No dropout and activation after upsampling.
    """

    def __init__(self, has_batch_norm=True, has_dropout=False,
                 has_activation_after_upsampling=False, name=None):
        super().__init__(has_batch_norm, has_dropout, has_activation_after_upsampling, name)

    def _build_model(self):
        input_layer = self._input()

        d0 = input_layer
        d0 = self._conv_k3_activation_possible_batchnorm(64)(d0)
        d0 = self._conv_k3_activation_possible_batchnorm(64)(d0)

        d1 = self._max_pooling()(d0)
        d1 = self._conv_k3_activation_possible_batchnorm(128)(d1)
        d1 = self._conv_k3_activation_possible_batchnorm(128)(d1)

        d2 = self._max_pooling()(d1)
        d2 = self._conv_k3_activation_possible_batchnorm(256)(d2)
        d2 = self._conv_k3_activation_possible_batchnorm(256)(d2)

        d3 = self._max_pooling()(d2)
        d3 = self._conv_k3_activation_possible_batchnorm(512)(d3)
        d3 = self._conv_k3_activation_possible_batchnorm(512)(d3)

        d3 = self._possible_dropout()(d3)

        d4 = self._max_pooling()(d3)
        d4 = self._conv_k3_activation_possible_batchnorm(1024)(d4)
        d4 = self._conv_k3_activation_possible_batchnorm(1024)(d4)

        d4 = self._possible_dropout()(d4)

        u3 = self._resize_conv(512)(d4)
        u3 = Concatenate()([d3, u3])
        u3 = self._conv_k3_activation_possible_batchnorm(512)(u3)
        u3 = self._conv_k3_activation_possible_batchnorm(512)(u3)

        u2 = self._resize_conv(256)(u3)
        u2 = Concatenate()([d2, u2])
        u2 = self._conv_k3_activation_possible_batchnorm(256)(u2)
        u2 = self._conv_k3_activation_possible_batchnorm(256)(u2)

        u1 = self._resize_conv(128)(u2)
        u1 = Concatenate()([d1, u1])
        u1 = self._conv_k3_activation_possible_batchnorm(128)(u1)
        u1 = self._conv_k3_activation_possible_batchnorm(128)(u1)

        u0 = self._resize_conv(64)(u1)
        u0 = Concatenate()([d0, u0])
        u0 = self._conv_k3_activation_possible_batchnorm(64)(u0)
        u0 = self._conv_k3_activation_possible_batchnorm(64)(u0)

        diff_layer = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                            kernel_initializer=self._conv_layer_initalizer,
                            kernel_regularizer=self._conv_layer_regualizer)(u0)

        output_layer = Add(name=DCAR_TargetAbstract.reconstruction_output_name) \
                        ([input_layer, diff_layer])

        model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        return model, input_layer, output_layer