import numpy as np

from LIDCArtifactReduction.generator.generator_transform import LIDCGeneratorNoisyTransform
from LIDCArtifactReduction.math_helpers.tensorflow_math_mixin import TensorflowMathMixin
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import RadonTransform


class IterativeARTResNetGeneratorTransform(TensorflowMathMixin, LIDCGeneratorNoisyTransform):
    def __init__(self, geometry: RadonGeometry, radon_transform: RadonTransform,
                 add_noise: bool, lnI0=5 * np.log(10), sum_scaling=5.0,
                 test_mode: bool = False):
        super().__init__(geometry=geometry, radon_transform=radon_transform,
                         add_noise=add_noise, lnI0=lnI0, sum_scaling=sum_scaling,
                         test_mode=test_mode)

    def transform(self, reconstructions, sinograms):
        # TODO: implement
        raise NotImplementedError()

    def _output_data_formatter(self, actual_reconstructions, bad_sinograms, good_reconstructions):
        return {IterativeARTResNet.imgs_input_name : actual_reconstructions,
                IterativeARTResNet.sinos_input_name: bad_sinograms,
                IterativeARTResNet.resnet_name: good_reconstructions}
