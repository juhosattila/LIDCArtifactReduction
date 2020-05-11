import numpy as np

from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.dicom_preprocess import DicomLoader
from LIDCArtifactReduction.radon_params import RadonParams
from LIDCArtifactReduction.offline_transformation import ResizeRescaleRadonOfflineTransformation


dl = DicomLoader(batch_size=5, ignored_edge_slice=0.1)
img_side_length = parameters.IMG_SIDE_LENGTH
radon_params = RadonParams(angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES))
offline_transformation = ResizeRescaleRadonOfflineTransformation(img_side_length, radon_params)
dl.run_offline_transformations(offline_transformation)


