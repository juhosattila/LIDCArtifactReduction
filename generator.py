import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import Iterator

from dicom_preprocess import ArrayStream

class LIDCDataGenerator:
    """Custom generator class for LIDC data.

    Data is assumed to be found in a directory passed to init.
    Capable of shuffling data, augmenting it with possible Poisson noise.
    It can provide iterator objects for training, validation and testing.
    """
    def __init__(self, validation_split=0.1, test_split=0.1, batch_size=16,
                 array_stream: ArrayStream = ArrayStream.instance()):
        self._batch_size = batch_size
        self._array_stream = array_stream

        filenames = self._array_stream.get_names()
        validtest_split = validation_split + test_split
        train_filenames, valid_test_filenames = train_test_split(filenames,
            train_size=1.0-validtest_split, shuffle=True)
        valid_filenames, test_filenames = train_test_split(valid_test_filenames,
            train_size=validation_split / validtest_split, shuffle=True)

        self._train_iterator = self.get_iterator(train_filenames, preprocess=True)
        self._valid_iterator = self.get_iterator(valid_filenames, preprocess=False)
        self._test_iterator = self.get_iterator(test_filenames, preprocess=False)

    def get_iterator(self, filenames, preprocess):
        return LIDCDataIterator(filenames, self._batch_size, preprocess,
                                self._array_stream)

    @property
    def train_iterator(self):
        return self._train_iterator

    @property
    def valid_iterator(self):
        return self._valid_iterator

    @property
    def test_iterator(self):
        return self._test_iterator


class LIDCDataIterator(Iterator):
    def __init__(self, filenames, batch_size, preprocess, array_stream: ArrayStream):
        self._filenames = filenames
        self._preprocess = preprocess
        self._array_stream = array_stream
        super(self).__init__(len(filenames), batch_size, shuffle=True, seed=None)

    def _get_batches_of_transformed_samples(self, index_array):
        """Get images with indices in index array, transform them.
        """
        # TODO: delete
        # indexek alapján kiválaszt jókat. Betölt. Transzformál, berendez kimenetet és visszaad
        goodrec_sino_pairs = [self._array_stream.load_arrays(filename)
                              for filename in self._filenames[index_array]]
        goodrecs = [pair[0] for pair in goodrec_sino_pairs]
        goodsinos = [pair[1] for pair in goodrec_sino_pairs]

        badrecs = [self.transform(goodsino) for goodsino in goodsinos ]

    def transform(self, sino):
        pass

    def generate_noise(self, sino):
        # TODO
        return sino




