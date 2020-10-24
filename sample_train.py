# Needed in this sequence
import LIDCArtifactReduction
#LIDCArtifactReduction.init()
LIDCArtifactReduction.init(gpu_memory_limit_MB=6700)

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_orig import ParallelRadonTransform
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.target_networks import DCAR_UNet_FewBatchNorms
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.DCAR_TrainingNetwork import DCAR_TrainingNetwork
from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.generator.generator import LIDCDataGenerator
from LIDCArtifactReduction.neural_nets.FBPPostProcessingResNet.FBPConvnet_generator_transform import FBPConvnetGeneratorTransform


geometry = RadonGeometry(volume_img_width=256, projection_width=256, nr_projections=40)
radon_transform = ParallelRadonTransform(geometry)

ds = DirectorySystem(geometry, data_name='skimage', algorithm_name='orig')
array_stream = RecSinoArrayStream(directory=ds.DATA_DIRECTORY)  # or ds.SAMPLE_DATA_DIRECTORY

generator = LIDCDataGenerator(array_stream=array_stream, load_data_config='data_config_file', verbose=True,
                            validation_split=0.25, test_split=0.25, batch_size=8)
# For noisy transformations you mat set lnI0 and sumscaling.
noisy_transformer = FBPConvnetGeneratorTransform(geometry, radon_transform,
                                                 add_noise=True)#, lnI0=10*np.log(4), sum_scaling=5.0)
train_iterator = generator.get_new_train_iterator(noisy_transformer)

non_noisy_transformer = FBPConvnetGeneratorTransform(geometry, radon_transform, add_noise=False)
validation_iterator = generator.get_new_validation_iterator(non_noisy_transformer)

unet = DCAR_UNet_FewBatchNorms(radon_geometry=geometry, name='UNet40-Sample', conv_regularizer=0.1 * 1e-3)
training_network = DCAR_TrainingNetwork(radon_geometry=geometry, radon_transformation=radon_transform,
                                        target_model=unet, name='UNet40_Training', dir_system=ds)

# Optionally load weights. latest=True loads the most recent weightfile and ignores 'name'.
#training_network.load_weights(name='UNet40_Sample_Training_weight_file')#latest=True)

# Setting training is always followed by compilation.
training_network.set_training(training=True)
training_network.compile(
    lr=1e-3, reconstruction_output_weight=10.0, sino_output_weight=1.0 / (16.0 * geometry.nr_projections),
    add_total_variation=True, total_variation_eps=0.001, tot_var_loss_weight=0.05, mse_tv_weight=3.0)
training_network.fit(train_iterator, validation_iterator, epochs=100,
                    steps_per_epoch=500, validation_steps=150,
                    verbose=1, initial_epoch=46, early_stoppig_patience=51)
