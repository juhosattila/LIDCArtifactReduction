import numpy as np

from LIDCArtifactReduction.math_helpers.math_mixin import MathMixin


class NumpyMathMixin(MathMixin):
    def shape(self, img):
        return np.shape(img)

    def log(self, x):
        return np.log(x)

    def exp(self, x):
        return np.exp(x)

    def random_normal(self, mean, stddev, size):
        return np.random.normal(mean, stddev, size)

    def random_poisson(self, mean):
        return np.random.poisson(mean)
