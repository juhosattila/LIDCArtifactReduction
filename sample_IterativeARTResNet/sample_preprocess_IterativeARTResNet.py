# Needed in this sequence
import LIDCArtifactReduction
LIDCArtifactReduction.init(gpu_id=0)
# LIDCArtifactReduction.init(gpu_id=0, gpu_memory_limit_MB=6700)

from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_orig import ForwardprojectionParallelRadonTransform
from LIDCArtifactReduction.dicom_preprocess import DicomLoader
from LIDCArtifactReduction.offline_transformation import ResizeRescaleRadonOfflineTransformation
from LIDCArtifactReduction.radon_transformation.radon_transformation_pyronn import \
    PyronnParallelForwardprojectionRadonTransform

# Some files

# patient_list = [1, 3, 5, 14]
# patient_list = range(400, 1020, 1)
# patient_ids = ["LIDC-IDRI-" + "{:04d}".format(id) for id in patient_list]
# dl = DicomLoader(batch_size=7, ignored_edge_slice=0.1).filter(patient_ids)

# All files
dl = DicomLoader(batch_size=7, ignored_edge_slice=0.1)

# ----------------------------

geometry = RadonGeometry(volume_img_width=256, projection_width=256, nr_projections=30)
# Here  change to corresponding algorithm. Now standard parallel and pyronn are available.
radon_transform = PyronnParallelForwardprojectionRadonTransform(geometry)
offline_transformation = ResizeRescaleRadonOfflineTransformation(geometry, radon_transform)

ds = DirectorySystem(geometry, data_name='pyronn', algorithm_name='IterativeARTResnet')
array_stream = RecSinoArrayStream(directory=ds.DATA_DIRECTORY)  # or ds.SAMPLE_DATA_DIRECTORY

dl.run_offline_transformations(offline_transformation, array_stream=array_stream, verbose=True)
