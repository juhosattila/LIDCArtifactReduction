import itertools
import os
from typing import List

import numpy as np

import parameters


class ArrayStream:
    def __init__(self, dir=None, array_names=None):
        self._base_dir = dir
        if self._base_dir is None:
            self._base_dir = parameters.DATA_DIRECTORY

        if not os.path.exists(self._base_dir):
            os.mkdir(self._base_dir)

        self._array_names = array_names
        self.actual_path = self._base_dir

    def save_arrays(self, name, arrays: List[np.ndarray]):
        file = os.path.join(self.actual_path, name)

        data_dict = dict()
        for key, arr in zip(self._array_names, arrays):
            data_dict[key] = arr

        np.savez(file=file, **data_dict)

    def get_directory_names(self):
        with os.scandir(self._base_dir) as dirit:
            result = [d.name for d in dirit if d.is_dir()]
        return result

    def _get_filenames(self, inner_path='.'):
        path = os.path.join(self._base_dir, inner_path)
        with os.scandir(path) as dirit:
            result = [os.path.join(inner_path, f.name) for f in dirit if f.is_file()]
        return result

    def get_names_with_dir(self, dir_or_dirs : str or List[str] = None):
        if dir_or_dirs is None:
            return self._get_filenames()

        dirs = dir_or_dirs
        if isinstance(dirs, str):
            dirs = [dirs]

        list_results = [self._get_filenames(direc) for direc in dirs]
        result = list(itertools.chain.from_iterable(list_results))
        return result

    def load_arrays(self, name_with_dir):
        file = os.path.join(self._base_dir, name_with_dir)
        result = []
        with np.load(file) as data:
            for key in self._array_names:
                result.append(data[key])
        return result

    def create_dir(self, dirname):
        path = os.path.join(self._base_dir, dirname)
        if not os.path.exists(path):
            os.mkdir(path)

    def switch_dir(self, dirname):
        self.create_dir(dirname)
        self.actual_path = os.path.join(self._base_dir, dirname)

    def unswitch(self):
        self.actual_path = self._base_dir

    _rec_sino_instance = None
    @classmethod
    def RecSinoInstance(cls):
        if cls._rec_sino_instance is None:
            cls._rec_sino_instance = ArrayStream(array_names=['rec', 'sino'])
        return cls._rec_sino_instance


class RecSinoArrayStream(ArrayStream):
    def __init__(self, dir=None):
        super().__init__(dir=dir, array_names=['rec', 'sino'])
