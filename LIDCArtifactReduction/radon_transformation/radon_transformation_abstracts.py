from abc import abstractmethod

from tensorflow_addons.utils.types import TensorLike


class ForwardprojectionRadonTransform:
    @abstractmethod
    def forwardproject(self, imgs: TensorLike):
        """
        Args:
            imgs: Array (numpy or tensorflow) in NHW or NHWC mode. Remember that H=W. C should be 1.
        Returns:
            Results is an NHWC array in numpy or tesnorflow based on implementation. Consider converting it.
        """
        pass


class BackprojectionRadonTransform:

    @abstractmethod
    def backproject(self, sinos: TensorLike):
        """
        Args:
            sinos: Array (numpy or tensorflow) in NHW or NHWC mode. C should be 1.
        Returns:
            Results is an NHWC array in numpy or tesnorflow based on implementation. Consider converting it.
        """
        pass


class RadonTransform(ForwardprojectionRadonTransform, BackprojectionRadonTransform):
    @abstractmethod
    def invert(self, sinos: TensorLike):
        """
        Args:
            sinos: Array (numpy or tensorflow) in NHW or NHWC mode. C should be 1.
        Returns:
            Results is an NHWC array in numpy or tesnorflow based on implementation. Consider converting it.
        """
        pass

class ARTRadonTransform:
    def __init__(self, alfa):
        self.alfa = alfa

    @abstractmethod
    def ART_step(self, imgs: TensorLike, sinos: TensorLike):
        """
        Args:
            imgs: Array (numpy or tensorflow) in NHW or NHWC mode. Remember that H=W. C should be 1.
            sinos: Array (numpy or tensorflow) in NHW or NHWC mode. C should be 1.
        Returns:
            Results is an NHWC array in numpy or tesnorflow based on implementation. Consider converting it.
        """
        pass
