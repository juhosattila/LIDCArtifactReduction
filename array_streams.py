import os
from typing import List

import numpy as np

import parameters


class ArrayStream:
    def __init__(self, dir=None, array_names=None):
        self.dir = dir
        if self.dir is None:
            self.dir = parameters.DATA_DIRECTORY

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self._array_names = array_names

    def save_arrays(self, name, arrays: List[np.ndarray]):
        file = os.path.join(self.dir, name)

        data_dict = dict()
        for key, arr in zip(self._array_names, arrays):
            data_dict[key] = arr

        np.savez(file=file, **data_dict)

    def get_names(self):
        return os.listdir(self.dir)

    def load_arrays(self, name):
        file = os.path.join(self.dir, name)
        result = []
        with np.load(file) as data:
            for key in self._array_names:
                result.append(data[key])
        return result

    _rec_sino_instance = None
    @classmethod
    def RecSinoInstance(cls):
        if cls._rec_sino_instance is None:
            cls._rec_sino_instance = ArrayStream(array_names=['rec', 'sino'])
        return cls._rec_sino_instance
