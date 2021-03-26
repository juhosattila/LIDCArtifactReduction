import numpy as np

from LIDCArtifactReduction.generator.generator_transform import LIDCGeneratorNoisyTransform
from LIDCArtifactReduction.math_helpers.numpy_math_mixin import NumpyMathMixin
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.DCAR_TrainingNetwork import DCAR_TrainingNetwork
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import RadonTransform


# TODO: make this usable with TensorflowMathMixin. Inject instead of inheriting. Inverting Radon is needed.
class FBPConvnetGeneratorTransform(NumpyMathMixin, LIDCGeneratorNoisyTransform):
    def __init__(self, geometry: RadonGeometry, radon_transform: RadonTransform,
                 add_noise: bool=True, lnI0=5 * np.log(10), sigma=0.3, sum_scaling=5.0,
                 test_mode: bool = False):
        super().__init__(geometry=geometry,
                         add_noise=add_noise, lnI0=lnI0, sigma=sigma, sum_scaling=sum_scaling,
                         test_mode=test_mode)
        self._radon_transform = radon_transform

    def transform(self, reconstructions, sinograms):
        noisy_sinograms = self.generate_sinogram_noise(sinograms) if self.add_noise else sinograms
        bad_reconstructions = self._radon_transform.invert(noisy_sinograms)

        if self._test_mode:
            return self._test_format_manager(bad_reconstructions, reconstructions, sinograms, noisy_sinograms)

        return DCAR_TrainingNetwork.input_data_encoder(
                    bad_reconstructions=bad_reconstructions,
                    bad_sinograms=noisy_sinograms,
                    good_reconstructions=reconstructions,
                    good_sinograms=sinograms)

        #return self._output_format_manager(bad_reconstructions, reconstructions, sinograms)

    # # TODO: probably should be provided directly by network. Similarly to all other spots.
    # def _output_format_manager(self, bad_reconstructions, good_reconstructions, good_sinograms):
    #     # TODO: if upgraded to TF 2.2 remove [None]
    #     # These 'None'-s correspond to weights attached to the outputs.
    #     return ({DCAR_TrainingNetwork.input_name : bad_reconstructions},
    #             {DCAR_TrainingNetwork.reconstruction_output_name : good_reconstructions,
    #              DCAR_TrainingNetwork.sino_output_name : good_sinograms}, [None, None])

    def _test_format_manager(self, bad_reconstructions, good_reconstructions, good_sinograms, bad_sinograms):
        return bad_reconstructions, [good_reconstructions, good_sinograms], bad_sinograms
