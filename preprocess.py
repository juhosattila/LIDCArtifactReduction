import numpy as np

import parameters
from dicom_preprocess import DicomLoader

from radon_transformation import RadonParams
from offline_transformation import ResizeRescaleRadonOfflineTransformation
from array_streams import RecSinoArrayStream


def run(dl: DicomLoader, dir):
    img_side_length = parameters.IMG_SIDE_LENGTH
    radon_params = RadonParams(angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES))
    offline_transformation = ResizeRescaleRadonOfflineTransformation(img_side_length, radon_params)
    array_stream = RecSinoArrayStream(dir)
    dl.run_offline_transformations(offline_transformation, array_stream,
                                   mode='patient')


def main():
    dl = DicomLoader(batch_size=5)
    run(dl, parameters.DATA_DIRECTORY)


def test():
    patient_list = [1, 3, 5, 14]
    patient_ids = ["LIDC-IDRI-" + "{:04d}".format(id) for id in patient_list]

    dl = DicomLoader(batch_size=2).filter(patient_ids)

    run(dl, parameters.TEST_DIRECTORY)

if __name__ == "__main__":
    #main()
    test()


