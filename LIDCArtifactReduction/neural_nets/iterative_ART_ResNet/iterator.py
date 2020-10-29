from typing import List

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.preprocessing.image import Iterator as KerasImgIterator

from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.data_formatter import output_data_formatter


class RecSinoArrayIterator(KerasImgIterator):
    def __init__(self, actual_reconstructions_batches: List[Tensor], sinograms_batches: List[Tensor],
                 good_reconstructions_batches: List[Tensor]):
        super().__init__(len(actual_reconstructions_batches), 1, shuffle=False, seed=None)
        self._actual_reconstructions_batches = actual_reconstructions_batches
        self._sinograms_batches = sinograms_batches
        self._good_reconstructions_batches = good_reconstructions_batches

    def _get_batches_of_transformed_samples(self, index_array):
        idx = index_array[0]
        return output_data_formatter(
            actual_reconstructions=tf.convert_to_tensor(self._actual_reconstructions_batches[idx], dtype=tf.float32),
            bad_sinograms=tf.convert_to_tensor(self._sinograms_batches[idx], dtype=tf.float32),
            good_reconstructions=tf.convert_to_tensor(self._good_reconstructions_batches[idx], dtype=tf.float32)
        )


class RecSinoSuperIterator(KerasImgIterator):
    def __init__(self, iterators):
        self._iterators = iterators
        super().__init__(len(iterators), batch_size=1, shuffle=True, seed=None)

    def _get_batches_of_transformed_samples(self, index_array):
        idx = index_array[0]
        return next(self._iterators[idx])
