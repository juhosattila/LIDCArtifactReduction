import numpy as np
from dicom_preprocess import DicomLoader, run_transformations
from tf_image import ParallelRadonTransform
import parameters
import os


def run():
    dl = DicomLoader(batch_size=5)
    radon_trans = ParallelRadonTransform(img_side_length=parameters.IMG_SIDE_LENGTH,
                    angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES))
    i = 0
    for data_batch in dl:
        data_batch_resized, data_batch_sino = run_transformations(data_batch, radon_trans)
        for rec, sino in zip(data_batch_resized, data_batch_sino):
            filename = '{:05}'.format(i)
            i += 1
            file = os.path.join(parameters.DATA_DIRECTORY, filename)
            np.savez(file=file, rec=rec, sino=sino)


if __name__ == "__main__":
    run()


