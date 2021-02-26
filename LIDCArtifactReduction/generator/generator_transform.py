from abc import abstractmethod

from LIDCArtifactReduction import parameters, utility

from LIDCArtifactReduction.math_helpers.math_mixin import MathMixin
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry


class LIDCGeneratorTransform:
    @abstractmethod
    def transform(self, reconstructions, sinograms):
        pass


class LIDCGeneratorNoisyTransform(LIDCGeneratorTransform, MathMixin):
    def __init__(self, geometry: RadonGeometry,
                 add_noise: bool, lnI0, sigma=0.3, sum_scaling=5.0, test_mode: bool=False):

        self._geometry = geometry

        self.add_noise = add_noise
        self.lnI0 = lnI0
        self.sum_scaling = sum_scaling
        # scaling of noise deviation parameter
        self.sigma = sigma

        self._test_mode = test_mode
        self._analysed = False

    @property
    def lnI0(self):
        return self._lnI0
    @lnI0.setter
    def lnI0(self, value):
        self._lnI0 = self.as_array(value)

    @property
    def sum_scaling(self):
        return self._sum_scaling

    @sum_scaling.setter
    def sum_scaling(self, value):
        self._sum_scaling = self.as_array(value)


    def generate_sinogram_noise(self, sino):
        """See documentation for explananation."""

        # pmax refers to the approximate maximum value in the sinogram
        # For scaling 1000HU to 1CT, we have (relevant sinomax) ~ IMG_SIDE_LENGTH
        pmax = self._geometry.volume_img_width * 1000.0 / parameters.HU_TO_CT_SCALING

        # Scaling defines how much the maximum sinogramvalue is related to lnI0.
        # Important: scaling is used on sinogram, so that value domains match with ln I0. At the end we invert scaling.
        # First there is mulitplication with 1/scale, at the very end multiplication with scale.
        # Remember, that the domain of the sinogram is chosen by us and not directly related to ln I0.
        scale = 1000.0 / parameters.HU_TO_CT_SCALING * self._geometry.volume_img_width / self.sum_scaling

        # deviation of noise
        sigma_I0 = self.as_array(self.sigma * self.exp(self.lnI0 - 1.0 / scale * pmax))

        I_normal_noise = self.random_normal(mean=0.0, stddev=sigma_I0, size=self.shape(sino))
        lnI = self.lnI0 - 1.0 / scale * sino
        I_no_noise = self.exp(lnI)

        I_added_noise = I_no_noise + I_normal_noise

        # some elements might be too low
        # too_low = I_added_noise < I_no_noise / 2.0
        # I_added_noise[too_low] = I_no_noise[too_low]
        I_added_noise = self.where(I_added_noise < I_no_noise / 2.0, I_no_noise, I_added_noise)

        I_Poisson = self.random_poisson(I_added_noise)

        lnI_added_noise_and_Poisson = self.log(I_Poisson)
        sino_added_noise = scale * (self.lnI0 - lnI_added_noise_and_Poisson)

        # Testing
        if self._test_mode and not self._analysed:
            self._analysed = True
            utility.analyse(I_no_noise, "I no noise")
            utility.analyse(I_added_noise, "I normal noise")
            utility.analyse(I_Poisson, "I Poisson")

        return sino_added_noise
