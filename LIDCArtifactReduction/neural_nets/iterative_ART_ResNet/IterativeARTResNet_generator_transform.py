import numpy as np
import tensorflow as tf
from LIDCArtifactReduction.generator.generator_transform import LIDCGeneratorNoisyTransform
from LIDCArtifactReduction.math_helpers.tensorflow_math_mixin import TensorflowMathMixin
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.data_formatter import output_data_formatter
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ARTRadonTransform


class IterativeARTResNetGeneratorTransform(TensorflowMathMixin, LIDCGeneratorNoisyTransform):
    def __init__(self, geometry: RadonGeometry, radon_transform: ARTRadonTransform,
                 mode: int = 4,
                 add_noise: bool = True, lnI0=10 * np.log(5), sum_scaling=5.0,
                 test_mode: bool = False):
        super().__init__(geometry=geometry,
                         add_noise=add_noise, lnI0=lnI0, sum_scaling=sum_scaling,
                         test_mode=test_mode)
        self._radon_transform = radon_transform
        self._mode = mode  # Later FBP could be added.

    def transform(self, reconstructions, sinograms):
        reconstructions_tf = tf.convert_to_tensor(reconstructions, dtype=tf.float32)
        bad_sinograms_tf = tf.convert_to_tensor(sinograms, dtype=tf.float32)
        bad_sinograms_tf = self.generate_sinogram_noise(bad_sinograms_tf) \
                            if self._add_noise else bad_sinograms_tf

        if self._test_mode:
            return reconstructions, sinograms, bad_sinograms_tf.numpy()

        actual_reconstructions_tf = tf.zeros_like(reconstructions_tf, dtype=tf.float32)
        for i in range(self._mode):
            actual_reconstructions_tf = self._radon_transform.ART_step(actual_reconstructions_tf, bad_sinograms_tf)

        return output_data_formatter(
            actual_reconstructions=actual_reconstructions_tf,
            bad_sinograms=bad_sinograms_tf,
            good_reconstructions=reconstructions_tf)
