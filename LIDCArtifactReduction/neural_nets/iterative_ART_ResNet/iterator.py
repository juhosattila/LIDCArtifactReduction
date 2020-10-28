from typing import List

from tensorflow.keras.preprocessing.image import Iterator as KerasImgIterator
from tensorflow_core import Tensor

from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNetTraining import \
    IterativeARTResNetTraningCustomTrainStepModel


class RecSinoArrayIterator(KerasImgIterator):
    def __init__(self, actual_reconstructions_batches: List[Tensor], sinograms_batches: List[Tensor],
                 good_reconstructions_batches: List[Tensor]):
        super().__init__(len(actual_reconstructions_batches), 1, shuffle=False, seed=None)
        self._actual_reconstructions_batches = actual_reconstructions_batches
        self._sinograms_batches = sinograms_batches
        self._good_reconstructions_batches = good_reconstructions_batches

    def _get_batches_of_transformed_samples(self, index_array):
        idx = index_array
        return IterativeARTResNetTraningCustomTrainStepModel.output_data_formatter(
            actual_reconstructions=self._actual_reconstructions_batches[idx],
            bad_sinograms=self._sinograms_batches[idx],
            good_reconstructions=self._good_reconstructions_batches[idx]
        )


class RecSinoSuperIterator(KerasImgIterator):
    def __init__(self, iterators):
        self._iterators = iterators
        super().__init__(len(iterators), batch_size=1, shuffle=True, seed=None)

    def _get_batches_of_transformed_samples(self, index_array):
        idx = index_array
        return next(self._iterators[idx])
