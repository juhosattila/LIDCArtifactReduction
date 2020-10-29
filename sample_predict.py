import LIDCArtifactReduction
#IDCArtifactReduction.init()
LIDCArtifactReduction.init(gpu_memory_limit_MB=6700)

from datetime import datetime

from LIDCArtifactReduction import utility, directory_system
from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.generator.generator import LIDCDataGenerator
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.DCAR_TrainingNetwork import DCAR_TrainingNetwork
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.target_networks import DCAR_UNet_FewBatchNorms
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.FBPConvnet_generator_transform import FBPConvnetGeneratorTransform
from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_orig import ParallelRadonTransform
from LIDCArtifactReduction.utility import show_grey


geometry = RadonGeometry(volume_img_width=256, projection_width=256, nr_projections=40)
radon_transform = ParallelRadonTransform(geometry)

ds = DirectorySystem(geometry, data_name='skimage', algorithm_name='orig')
array_stream = RecSinoArrayStream(directory=ds.DATA_DIRECTORY)  # or ds.SAMPLE_DATA_DIRECTORY

generator = LIDCDataGenerator(array_stream=array_stream, data_configuration_dir=ds.DATA_CONFIGURATION_DIR,
                              load_data_config='data_config_file',  # or True
                              verbose=True, batch_size=8, shuffle=True)

# For noisy transformations you mat set lnI0 and sumscaling.
noisy_transformer = FBPConvnetGeneratorTransform(geometry, radon_transform,
                                                 add_noise=True)#, lnI0=10*np.log(4), sum_scaling=5.0)
test_iterator = generator.get_new_test_iterator(noisy_transformer)

unet = DCAR_UNet_FewBatchNorms(radon_geometry=geometry, name='UNet_L1TV_ValRec_Sample', conv_regularizer=0.5 * 1e-3)
training_network = DCAR_TrainingNetwork(radon_geometry=geometry, radon_transformation=radon_transform,
                                        target_model=unet, name='UNet_L1TV_ValRec_Sample_Training',
                                        dir_system=ds)
training_network.load_weights(name='UNet_Transferred_NoTV_Training.06-0.0042')#latest=True)
training_network.set_inference_mode()

test_batch = next(test_iterator)
test_batch = test_batch[:2]
results = training_network(test_batch)
results = (results[0].numpy(), results[1].numpy())

nr_of_samples = 3
timestemp = datetime.now().strftime("%Y%m%d-%H%M%S")
base_dir = utility.direc(ds.PREDICTED_IMAGES_DIR, 'UNet_Transferred_NoTV_Training.06-0.0042' )

for i in range(nr_of_samples):

    direc = utility.direc( base_dir, '{:02d}'.format(i))
    show_grey([
        test_batch[0][DCAR_TrainingNetwork.input_name][i],
        test_batch[1][DCAR_TrainingNetwork.reconstruction_output_name][i],
        results[0][i]
        ], norm_values=[-1.0, 5.0],
        save_names=['in_noisy_rec', 'out_expected_rec', 'predicted_rec'], directory=direc)

    show_grey([
        test_batch[1][DCAR_TrainingNetwork.sino_output_name][i],
        results[1][i]
        ], norm_values=[-5.0, 300.0],
        save_names=['out_expected_sino', 'predicted_sino'], directory=direc)

    show_grey([
        test_batch[1][DCAR_TrainingNetwork.reconstruction_output_name][i] - results[0][i],
        ], norm_values=[-10.0, 10.0],
        save_names=['expected_predicted_rec_diff'], directory=direc)
    show_grey([
        test_batch[1][DCAR_TrainingNetwork.sino_output_name][i] - results[1][i]
        ], norm_values=[-255.0, 255.0],
        save_names=['expected_predicted_sino_diff'], directory=direc)

