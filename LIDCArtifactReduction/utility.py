import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Iterable, List
import itertools

import tensorflow as tf


def _process_tensorlike(img_or_imgs: Union[tf.Tensor, np.ndarray]) -> List[np.ndarray]:  # HWx or NHWx
    processed_imgs = img_or_imgs
    if isinstance(processed_imgs, tf.Tensor):
        processed_imgs = processed_imgs.numpy()

    shape = np.shape(processed_imgs)
    if shape[-1] == 1:
        processed_imgs = np.reshape(processed_imgs, newshape=shape[:-1])  # HW or NHW

    if np.ndim(processed_imgs) == 2:  # HW
        processed_imgs = [processed_imgs]
    else:  # NHW
        processed_imgs = [img for img in processed_imgs]

    return processed_imgs


def show_grey(img_or_imgs: Union[tf.Tensor, np.ndarray, Iterable], norm_values=None, save_names=None, directory=None):
    """Multifunctional image displaying function with possible option to save the displayed image to a specified
    directory.

    Args:
        img_or_imgs: should be an image (HWx) or batch of images (NHWx)
            or Iterable of images (e.g. list of tensors, images, that are HWx).
            They can be tensorflow or numpy images. If enumerable, then each of them of arbitrary size.
            Images should not have channel information, only if the channel dimension has size 1.
            Hence x is either None or if C, than C=1.
            W must be more than 1, if there is no channel information.

        norm_values: if provided, it should be a tuple (min, max)
        save_names: if provided, it should be an array, of names
    """
    processed_imgs = img_or_imgs
    if isinstance(processed_imgs, (tf.Tensor, np.ndarray)):  # HWx or NHWx
        processed_imgs = _process_tensorlike(processed_imgs)  # List of HW
    else:  # Iterable
        list_of_img_lists = [_process_tensorlike(img) for img in processed_imgs]
        processed_imgs = itertools.chain.from_iterable(list_of_img_lists)

    cmap = 'gray'
    normargs = {'vmin':norm_values[0], 'vmax':norm_values[1]} if norm_values is not None else {}

    for idx, img in enumerate(processed_imgs):
        plt.imshow(img, cmap=cmap, **normargs)
        if save_names is not None:
            filename = os.path.join(directory, save_names[idx] + '.png')
            plt.imsave(arr=img, fname=filename, cmap=cmap, **normargs)
            # print(filename)
        plt.show()


def analyse(arrs, names=None):
    """Analyse list of numpy arrays in the form of HW, NHW or NHWC
    """
    arrays = np.asarray(arrs, dtype='float32')
    if np.ndim(arrays) == 2:
        arrays = [arrays]

    def analyse_one(arr, name=None):
        print("-------------------------")
        print(f"Analysing array {name}:")
        print("Min: ", np.min(arr))
        print("Max: ", np.max(arr))
        print("-------------------------")

        show_grey(arr)
        plt.hist(arr)
        plt.show()

    if names is None:
        for arr in arrays:
            analyse_one(arr)

    else:
        if not isinstance(names, list):
            names = [names]
        for arr, name in zip(arrays, names):
            analyse_one(arr, name)


def direc(*args):
    """Join the parameters into a directory path and create it if necessary."""
    direc_path = os.path.join(*args)
    if not os.path.exists(direc_path):
        os.makedirs(direc_path)
    return direc_path


class ProgressNumber:
    def __init__(self, max_value):
        self._max_value = max_value
        print("Progress: [{:5d} / {:5d}]".format(0, self._max_value))
        self._actual = 0

    def update_add(self, i):
        self._actual += i
        print("Progress: [{:5d} / {:5d}]".format(self._actual, self._max_value))


def get_filepath(name=None, directory=None, latest=False, extension=''):
    """Returns filepath in case of different scenarios.

    We either do not know if it is full or relative path, but know the directory. Or we wish to get the
    full path of the latest file with a certain extension.

    Args:
        name: filename of interest. Could have full path or not. Extension will be attached,
            if it does not already have one.
        latest: If true and there is a file in the given directory, then the latest of these is loaded,
            ignoring argument 'name'. If there is not a file, than 'name' is loaded.
        extension: It may or may not contain '.' .
    """
    success = False
    extension_with_point = '.' + extension.lstrip('.')
    if latest:
        list_of_files = glob.glob(os.path.join(directory, '*' + extension_with_point))
        # check emptiness, correctness: https://docs.python.org/3/library/stdtypes.html#truth-value-testing
        if list_of_files:
            filepath = max(list_of_files, key=os.path.getctime)
            success = True

    if not success:
        filepath = name
        if directory:
            name_no_base = os.path.basename(name)
            filepath = os.path.join(directory, name_no_base)
        if not filepath.endswith(extension_with_point):
            filepath = filepath + extension_with_point

    return filepath





# TODO: implement logging of fit data
# ----------------------------- TESTING

class Logger:
    def log(self, message):
        pass

    def newline(self):
        self.log("\n")

    def log_newline(self, message):
        self.log(message)
        self.newline(self)

    def log_value(self, message):
        pass

    def close(self):
        pass


class FileLogger(Logger):
    def __init__(self, log_file_name):
        if log_file_name is None or log_file_name == "":
            raise Exception("Bad string received.")

        self.log_file_name = log_file_name
        self.logs_folder = 'Logs/'
        self.log_file_path = self.log_file_name + ".log"
        self.log_file_path = os.path.join(self.logs_folder, self.log_file_path)
        self.log_file = open(self.log_file_path, mode="a+", newline='')

    def log(self, message):
        self.log_file.write(message)
        # So that interrupt won't delete full content.
        self.log_file.flush()

    def close(self):
        self.log_file.close()
