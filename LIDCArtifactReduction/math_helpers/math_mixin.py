from abc import abstractmethod


class MathMixin:
    @abstractmethod
    def shape(self, img):
        pass

    @abstractmethod
    def log(self, x):
        pass

    @abstractmethod
    def exp(self, x):
        pass

    @abstractmethod
    def random_normal(self, mean, stddev, size):
        pass

    @abstractmethod
    def random_poisson(self, mean):
        pass
