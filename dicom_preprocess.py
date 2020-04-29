import parameters
import pylidc as pl
import numpy as np
import tensorflow as tf
from tf_image import scale_HU2Radio

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


# TODO
#@tf.function
def run_transformations(data, radon_trans):
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    data_tf = tf.expand_dims(data_tf, axis=-1)
    resized_data = tf.image.resize(data_tf,
                        size=[parameters.IMG_SIDE_LENGTH, parameters.IMG_SIDE_LENGTH])
    scaled_data = scale_HU2Radio(resized_data)
    data_sino = radon_trans(scaled_data)
    return scaled_data.numpy(), data_sino.numpy()
