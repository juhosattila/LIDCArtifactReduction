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


def show_grey(img_or_imgs: Union[tf.Tensor, np.ndarray, Iterable]):
    """
    :param img_or_imgs: should be an image (HWx) or batch of images (NHWx)
        or Iterable of images (e.g. list of tensors, images, that are HWx).
        They can be tensorflow or numpy images. If enumerable, then each of them of arbitrary size.
        Images should not have channel information, only if the channel dimension has size 1.
        Hence x is either None or if C, than C=1.
        W must be more than 1, if there is no channel information.
    """
    processed_imgs = img_or_imgs
    if isinstance(processed_imgs, (tf.Tensor, np.ndarray)):  # HWx or NHWx
        processed_imgs = _process_tensorlike(processed_imgs)  # List of HW
    else:  # Iterable
        list_of_img_lists = [_process_tensorlike(img) for img in processed_imgs]
        processed_imgs = itertools.chain.from_iterable(list_of_img_lists)

    for img in processed_imgs:
        plt.imshow(img, cmap=plt.cm.Greys_r)
        #plt.imshow(img, cmap=plt.get_cmap('Greys'))
        plt.show()


def analyse(arrs, names=None):
    """Analyse list of numpy arrays in the form, of HW, NHW or NHWC
    """
    arrays = np.asanyarray(arrs, dtype='float32')
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
