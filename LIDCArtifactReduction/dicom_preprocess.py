from typing import List
import numpy as np

import pylidc as pl

from LIDCArtifactReduction.offline_transformation import DicomOfflineTransformation
from LIDCArtifactReduction.array_streams import ArrayStream


class ScanWrapper:
    def __init__(self, scan: pl.Scan):
        self._scan = scan

        self._data_set = False
        self._volume = None
        self._intercepts = None
        self._slopes = None

        self._dtype = np.float32

    def _set_data(self, verbose=False):
        if self._data_set:
            return

        images = self._scan.load_all_dicom_images(verbose=verbose)
        self._volume = np.stack([img.pixel_array for img in images], axis=0).astype(self._dtype)
        self._intercepts = np.array([img.RescaleIntercept for img in images], dtype=self._dtype)
        self._slopes = np.array([img.RescaleSlope for img in images]).astype(self._dtype)
        self._data_set = True

    @property
    def patient_id(self):
        return self._scan.patient_id

    @property
    def volume(self):
        self._set_data()
        return self._volume

    @property
    def intercepts(self):
        self._set_data()
        return self._intercepts

    @property
    def slopes(self):
        self._set_data()
        return self._slopes

    def __len__(self):
        self._set_data()
        return len(self._volume)


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
        self._scanw_list = [ScanWrapper(scan) for scan in self._scan.all()]
        return self

    def __next__(self) -> List[ScanWrapper]:
        if self._actual_element >= len(self._scanw_list):
            raise StopIteration

        after_last_element = min(self._actual_element + self.batch_size, len(self._scanw_list))
        relevant_scan_list = self._scanw_list[self._actual_element: after_last_element]
        self._actual_element = after_last_element
        return relevant_scan_list

    def run_offline_transformations(self, offline_transformation: DicomOfflineTransformation,
                                    array_stream = ArrayStream.RecSinoInstance(),
                                    mode = 'patient'):
        """
        Args:
             mode: one of 'patient' or 'combined'
        """
        if mode == 'combined':
            i = 0
            for scanw_batch in self:
                volumes = np.concatenate([scanw.volume for scanw in scanw_batch], axis=0)
                intercepts = np.concatenate([scanw.intercepts for scanw in scanw_batch], axis=0)
                slopes = np.concatenate([scanw.slopes for scanw in scanw_batch], axis=0)

                output_data_batch = offline_transformation(volumes, intercepts=intercepts, slopes=slopes)
                for output_data in output_data_batch:
                    filename = '{:05}'.format(i)
                    i += 1
                    array_stream.save_arrays(filename, output_data)

        else: # mode == 'patient'

            for scanw_batch in self:
                volumes = np.concatenate([scanw.volume for scanw in scanw_batch], axis=0)
                intercepts = np.concatenate([scanw.intercepts for scanw in scanw_batch], axis=0)
                slopes = np.concatenate([scanw.slopes for scanw in scanw_batch], axis=0)

                patient_ids = [scanw.patient_id for scanw in scanw_batch]
                nrs_imgs = [len(scanw) for scanw in scanw_batch]
                imgs_boundaries = [sum(nrs_imgs[:i+1]) for i in range(len(nrs_imgs)-1)]

                output_data_batch = offline_transformation(volumes, intercepts=intercepts, slopes=slopes)
                patient_data_batch = [output_data_batch[i:j]
                                      for i,j in zip([0] + imgs_boundaries, imgs_boundaries + [None])]

                for patient_id, patient_data in zip(patient_ids, patient_data_batch):
                    array_stream.switch_dir(patient_id)
                    for id, img_data in enumerate(patient_data):
                        arrname = '{:04}'.format(id)
                        array_stream.save_arrays(arrname, img_data)