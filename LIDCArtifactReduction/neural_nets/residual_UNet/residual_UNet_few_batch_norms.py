from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, \
    Concatenate, Add

from LIDCArtifactReduction.neural_nets.residual_UNet.residual_UNet_abstract import ResidualUNetAbstract


# TODO: decide whether dropout and batchnorm are necessary
class ResidualUNetFewBatchNorms(ResidualUNetAbstract):
    """Defaults are made with some changes based on the article
    Huang, Wurfle: Some investigations on Robustness of Deep Learning in Limited Andle Tomography (2018).
    This means that by default there is batch norm, but not everywhere,
    there is NO dropout and activation after upsampling.
    """
    def __init__(self, volume_img_width: int, has_batch_norm=True, has_dropout=False,
                 has_activation_after_upsampling=False, conv_regularizer=None,
                 name=None, input_name=None, output_name=None,
                 output_difference_layer: bool = False, difference_name=None):
        super().__init__(volume_img_width, has_batch_norm, has_dropout, has_activation_after_upsampling,
                         conv_regularizer=conv_regularizer, name=name, input_name=input_name, output_name=output_name,
                         output_difference_layer=output_difference_layer, difference_name=difference_name)

    def _build_model(self):
        input_layer = self._input()

        d0 = input_layer
        d0 = self._conv_k3_activation(64)(d0)
        d0 = self._conv_k3_activation(64)(d0)

        d1 = self._max_pooling()(d0)
        d1 = self._possible_batch_norm()(d1)
        d1 = self._conv_k3_activation(128)(d1)
        d1 = self._conv_k3_activation(128)(d1)

        d2 = self._max_pooling()(d1)
        d2 = self._possible_batch_norm()(d2)
        d2 = self._conv_k3_activation(256)(d2)
        d2 = self._conv_k3_activation(256)(d2)

        d3 = self._max_pooling()(d2)
        d3 = self._possible_batch_norm()(d3)
        d3 = self._conv_k3_activation(512)(d3)
        d3 = self._conv_k3_activation(512)(d3)

        d3 = self._possible_dropout()(d3)

        d4 = self._max_pooling()(d3)
        d4 = self._possible_batch_norm()(d4)
        d4 = self._conv_k3_activation(1024)(d4)
        d4 = self._conv_k3_activation(1024)(d4)

        d4 = self._possible_dropout()(d4)

        u3 = self._resize_conv(512)(d4)
        u3 = Concatenate()([d3, u3])
        u3 = self._possible_batch_norm()(u3)
        u3 = self._conv_k3_activation(512)(u3)
        u3 = self._conv_k3_activation(512)(u3)

        u2 = self._resize_conv(256)(u3)
        u2 = Concatenate()([d2, u2])
        u2 = self._possible_batch_norm()(u2)
        u2 = self._conv_k3_activation(256)(u2)
        u2 = self._conv_k3_activation(256)(u2)

        u1 = self._resize_conv(128)(u2)
        u1 = Concatenate()([d1, u1])
        u1 = self._possible_batch_norm()(u1)
        u1 = self._conv_k3_activation(128)(u1)
        u1 = self._conv_k3_activation(128)(u1)

        u0 = self._resize_conv(64)(u1)
        u0 = Concatenate()([d0, u0])
        u0 = self._possible_batch_norm()(u0)
        u0 = self._conv_k3_activation(64)(u0)
        u0 = self._conv_k3_activation(64)(u0)

        difference_layer = self._difference_layer(u0)
        output_layer = Add(name=self._output_name)([input_layer, difference_layer])

        if self._output_difference_layer:
            model = Model(inputs=input_layer, outputs=[output_layer, difference_layer], name=self.name)
        else:
            model = Model(inputs=input_layer, outputs=output_layer, name=self.name)

        self._model = model
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._difference_layer = difference_layer
