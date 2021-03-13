from typing import List
import numpy as np
from tensorflow.keras.utils import Progbar

import pylidc as pl

from LIDCArtifactReduction.offline_transformation import DicomOfflineTransformation
from LIDCArtifactReduction.array_streams import ArrayStream
from LIDCArtifactReduction import utility
from LIDCArtifactReduction.utility import ProgressNumber


class ScanWrapper:
    def __init__(self, scan: pl.Scan, ignored_front_slice, ignored_end_slice):
        self._scan = scan
        self._ignored_front_slice = ignored_front_slice
        self._ignored_end_slice = ignored_end_slice
        self._dtype = np.float32

    def load_data(self, verbose=False):
        images = self._scan.load_all_dicom_images(verbose=verbose)

        nr_ignored_front = np.math.floor(len(images) * self._ignored_front_slice)
        nr_ignored_end = np.math.floor(len(images) * self._ignored_end_slice)
        images = images[nr_ignored_front: -nr_ignored_end]

        volume = np.stack([img.pixel_array for img in images], axis=0).astype(self._dtype)
        intercepts = np.array([img.RescaleIntercept for img in images], dtype=self._dtype)
        slopes = np.array([img.RescaleSlope for img in images]).astype(self._dtype)

        return {'len': len(volume), 'volume': volume, 'intercepts': intercepts, 'slopes': slopes}

    @property
    def patient_id(self):
        return self._scan.patient_id


class DicomLoader:
    def __init__(self, batch_size=10, ignored_front_slice=0.30, ignored_end_slice=0.15):
        """
        Note that the edges of volumes contain a lot of irrelevant slices that cannot be thouroughly examinded
        using X-ray CT modality. Hence it is possible to cut them off. In the dicom volumes around 15% slices at the
        front and around 30% slices at the end are irrelevant. But, the scanning changes the order of slices. Hence
        30% needs to be cut of from the front and around 15% from the end.

        Args:
            ignored_front_slice: percentage of slices ignored at the front of each volume. Around 30% are irrelevant.
            ignored_end_slice: percentage of slices ignored at the end of each volume. Around 15% are irrelevant.
        """
        self._batch_size = batch_size
        self._scan = pl.query(pl.Scan)
        self._ignored_front_slice = ignored_front_slice
        self._ignored_end_slice = ignored_end_slice

        self._scan_list_created = False

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

    def _create_scan_list(self):
        if not self._scan_list_created:
            self._scanw_list = [ScanWrapper(scan, self._ignored_front_slice, self._ignored_end_slice) for scan in self._scan.all()]
        self._scan_list_created = True

    def __len__(self):
        self._create_scan_list()
        return len(self._scanw_list)

    def __iter__(self):
        self._actual_element = 0
        self._create_scan_list()
        return self

    def __next__(self) -> List[ScanWrapper]:
        if self._actual_element >= len(self._scanw_list):
            raise StopIteration

        after_last_element = min(self._actual_element + self.batch_size, len(self._scanw_list))
        relevant_scan_list = self._scanw_list[self._actual_element: after_last_element]
        self._actual_element = after_last_element
        return relevant_scan_list

    def run_offline_transformations(self, offline_transformation: DicomOfflineTransformation,
                                    array_stream: ArrayStream, verbose=True):

        if verbose:
            print("We are starting offline transformation:")
            #progress: ProgressNumber = utility.ProgressNumber(max_value=len(self))
            progbar = Progbar(len(self))

        for scanw_batch in self:
            patient_ids = [scanw.patient_id for scanw in scanw_batch]
            data_batch = [scanw.load_data() for scanw in scanw_batch]

            nrs_imgs = [data['len'] for data in data_batch]
            volumes = np.concatenate([data['volume'] for data in data_batch], axis=0)
            intercepts = np.concatenate([data['intercepts'] for data in data_batch], axis=0)
            slopes = np.concatenate([data['slopes'] for data in data_batch], axis=0)

            imgs_boundaries = [sum(nrs_imgs[:i+1]) for i in range(len(nrs_imgs)-1)]

            output_data_batch = offline_transformation(volumes, intercepts=intercepts, slopes=slopes)
            patient_data_batch = [output_data_batch[i:j]
                                  for i, j in zip([0] + imgs_boundaries, imgs_boundaries + [None])]

            for patient_id, patient_data in zip(patient_ids, patient_data_batch):
                array_stream.switch_dir(patient_id)
                for idex, img_data in enumerate(patient_data):
                    arrname = '{:04}'.format(idex)
                    array_stream.save_arrays(arrname, img_data)

            if verbose:
                #progress.update_add(len(scanw_batch))
                progbar.add(1)
