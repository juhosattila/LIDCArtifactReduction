import pylidc as pl
import numpy as np
from skimage.exposure import rescale_intensity
import os

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike

FILEDIR = 'images'
FILENAME = 'sample_imgs.npz'
IMG_ARRAY_NAME = "imgs"

file = os.path.join(FILEDIR, FILENAME)


def save_imgs(imgs):
    np.savez_compressed(file, **{IMG_ARRAY_NAME: imgs})


def sample_preprocess_dicom():
    pid = 'LIDC-IDRI-0001'
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = np.transpose(scan.to_volume(), [2, 0, 1])
    # one_pic = vol[55]
    multiple_pics = vol[32:64]

    # pic_rescaled = rescale_intensity(one_pic, out_range=(0.0, 1.0))
    imgs_rescaled = rescale_intensity(multiple_pics, out_range=(0.0, 1.0))
    save_imgs(imgs_rescaled)


def sample_load_imgs():
    with np.load(file) as data:
        sample_img = data[IMG_ARRAY_NAME]
    return sample_img





def my_to_volume(scan, verbose=False):
    images = scan.load_all_dicom_images(verbose=verbose)
    volume = np.stack([img.pixel_array for img in images], axis=0).astype('float32')
    return volume


class DicomLoader():
    def __init__(self, batch_size=10):
        self._batch_size = batch_size
        self._scan = pl.query(pl.Scan)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def filter(self, patient_ids=None):
        if patient_ids is not None:
            self._scan = self._scan.filter(pl.Scan.patient_id.in_(patient_ids))
        return self

    def __iter__(self):
        self._actual_element = 0
        self._scan_list = self._scan.all()
        return self

    def __next__(self):
        if self._actual_element >= len(self._scan_list):
            raise StopIteration

        after_last_element = min(self._actual_element + self.batch_size, len(self._scan_list))
        relevant_scan_list = self._scan_list[self._actual_element: after_last_element]
        array_list = [my_to_volume(scan) for scan in relevant_scan_list]
        array = np.concatenate(array_list, axis=0)

        self._actual_element = after_last_element

        return array


