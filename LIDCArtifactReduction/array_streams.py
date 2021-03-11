import itertools
import os
from typing import List

import numpy as np


class ArrayStream:
    def __init__(self, directory, array_names=None):
        self._base_dir = directory

        if not os.path.exists(self._base_dir):
            os.mkdir(self._base_dir)

        self._array_names = array_names
        self._actual_path = self._base_dir

    def save_arrays(self, name, arrays: List[np.ndarray]):
        file = os.path.join(self._actual_path, name)

        # change to following is correct and more concise:
        # data_dict = dict(zip(self._array_names, arrays))
        data_dict = dict()
        for key, arr in zip(self._array_names, arrays):
            data_dict[key] = arr

        np.savez(file=file, **data_dict)

    def get_directory_names(self):
        with os.scandir(self._base_dir) as dirit:
            result = [d.name for d in dirit if d.is_dir()]
        result.sort()
        return result

    def _get_filenames(self, inner_path='.'):
        path = os.path.join(self._base_dir, inner_path)
        with os.scandir(path) as dirit:
            result = [os.path.join(inner_path, f.name) for f in dirit if f.is_file()]
        result.sort()
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
        """
        Args:
             name_with_dir: Name attached with inner directory relative to the base directory.
        """
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
        self._actual_path = os.path.join(self._base_dir, dirname)

    def unswitch(self):
        self._actual_path = self._base_dir


class RecSinoArrayStream(ArrayStream):
    def __init__(self, directory):
        super().__init__(directory, array_names=['rec', 'sino'])
