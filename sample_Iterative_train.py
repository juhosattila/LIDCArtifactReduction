# Needed in this sequence
import LIDCArtifactReduction
LIDCArtifactReduction.init()
#LIDCArtifactReduction.init(gpu_memory_limit_MB=6700)

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.generator.generator import LIDCDataGenerator
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNetTraining import IterativeARTResNetTraining
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet_generator_transform import \
    IterativeARTResNetGeneratorTransform
from LIDCArtifactReduction.radon_transformation.radon_transformation_pyronn import PyronnParallelARTRadonTransform


geometry = RadonGeometry(volume_img_width=256, projection_width=256, nr_projections=40)
radon_transform = PyronnParallelARTRadonTransform(geometry, alfa=0.5 / 256)

ds = DirectorySystem(geometry, data_name='pyronn', algorithm_name='IterativeARTResnet')
array_stream = RecSinoArrayStream(directory=ds.DATA_DIRECTORY)  # or ds.SAMPLE_DATA_DIRECTORY

generator = LIDCDataGenerator(array_stream=array_stream, data_configuration_dir=ds.DATA_CONFIGURATION_DIR,
                              load_data_config='config20201028-031556', verbose=True,
                            validation_split=0.1, test_split=0.1, batch_size=8)
# For noisy transformations you may set lnI0 and sumscaling.
noisy_transformer = IterativeARTResNetGeneratorTransform(geometry, radon_transform, add_noise=True
                                                         )#mode=4, #, lnI0=10*np.log(4), sum_scaling=5.0)
train_iterator = generator.get_new_train_iterator(noisy_transformer)

# non_noisy_transformer = FBPConvnetGeneratorTransform(geometry, radon_transform, add_noise=False)
# validation_iterator = generator.get_new_validation_iterator(non_noisy_transformer)

resnet = IterativeARTResNet(radon_geometry=geometry, radon_transformation=radon_transform,
                             name='Sample_IterativeARTResNet')
training_network = IterativeARTResNetTraining(radon_transformation=radon_transform,
                                        target_model=resnet, name='Sample_IterativeARTResNet_Training')#, dir_system=ds)

# Optionally load weights. latest=True loads the most recent weightfile and ignores 'name'.
#training_network.load_weights(name='Sample_IterativeARTResNet_Training')#latest=True)


training_network.compile(
    lr=1e-3,
    reconstructions_output_weight=1.0,
    error_singrom_weight=1.0 / (1.0 * geometry.volume_img_width),
    gradient_weight=1.0)
training_network.train(train_iterator, final_depth=5, steps_per_epoch=10)
