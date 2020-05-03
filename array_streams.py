import os
from typing import List

import numpy as np

import parameters


class ArrayStream:
    def __init__(self, dir=None, array_names=None):
        self.base_dir = dir
        if self.base_dir is None:
            self.base_dir = parameters.DATA_DIRECTORY

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self._array_names = array_names
        self.actual_path = self.base_dir

    def save_arrays(self, name, arrays: List[np.ndarray]):
        file = os.path.join(self.actual_path, name)

        data_dict = dict()
        for key, arr in zip(self._array_names, arrays):
            data_dict[key] = arr

        np.savez(file=file, **data_dict)

    def get_names(self):
        return os.listdir(self.base_dir)

    def load_arrays(self, name):
        file = os.path.join(self.base_dir, name)
        result = []
        with np.load(file) as data:
            for key in self._array_names:
                result.append(data[key])
        return result

    def create_dir(self, dirname):
        path = os.path.join(self.base_dir, dirname)
        if not os.path.exists(path):
            os.mkdir(path)

    def switch_dir(self, dirname):
        self.create_dir(dirname)
        self.actual_path = os.path.join(self.base_dir, dirname)

    def unswitch(self):
        self.actual_path = self.base_dir

    _rec_sino_instance = None
    @classmethod
    def RecSinoInstance(cls):
        if cls._rec_sino_instance is None:
            cls._rec_sino_instance = ArrayStream(array_names=['rec', 'sino'])
        return cls._rec_sino_instance


class RecSinoArrayStream(ArrayStream):
    def __init__(self, dir=None):
        super().__init__(dir=dir, array_names=['rec', 'sino'])
