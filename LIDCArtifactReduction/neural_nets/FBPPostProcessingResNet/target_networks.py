from abc import abstractmethod

from LIDCArtifactReduction.neural_nets.ModelInterface import ModelInterface
from LIDCArtifactReduction.neural_nets.residual_UNet.residual_UNet_few_batch_norms import ResidualUNetFewBatchNorms
from LIDCArtifactReduction.neural_nets.residual_UNet.residual_UNet_many_batch_norms import ResidualUNetManyBatchNorms
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry


class DCAR_TargetInterface(ModelInterface):
    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def input_layer(self):
        pass

    @property
    @abstractmethod
    def output_layer(self):
        pass

    input_name = 'input_layer'
    reconstruction_output_name = 'reconstruction_output_layer'

# TODO: decide whether dropout and batchnorm are necessary
class DCAR_UNet_FewBatchNorms(DCAR_TargetInterface, ResidualUNetFewBatchNorms):
    def __init__(self, radon_geometry: RadonGeometry, has_batch_norm=True, has_dropout=False,
                 has_activation_after_upsampling=False, conv_regularizer=None, name=None):
        super().__init__(radon_geometry.volume_img_width, has_batch_norm, has_dropout, has_activation_after_upsampling,
                         conv_regularizer=conv_regularizer,
                         name=name,
                         input_name=DCAR_TargetInterface.input_name,
                         output_name=DCAR_TargetInterface.reconstruction_output_name)


class DCAR_UNet_ManyBatchNorms(DCAR_TargetInterface, ResidualUNetManyBatchNorms):
    """Defaults are made according to article
    Huang, Wurfle: Some investigations on Robustness of Deep Learning in Limited Andle Tomography (2018).
    This means that by default there is batch norm, there is No dropout and activation after upsampling.
    """
    def __init__(self, radon_geometry: RadonGeometry, has_batch_norm=True, has_dropout=False,
                 has_activation_after_upsampling=False, conv_regularizer=None, name=None):
        super().__init__(radon_geometry.volume_img_width, has_batch_norm, has_dropout, has_activation_after_upsampling,
                         conv_regularizer=conv_regularizer,
                         name=name,
                         input_name=DCAR_TargetInterface.input_name,
                         output_name=DCAR_TargetInterface.reconstruction_output_name)
