# Needed in this sequence
import LIDCArtifactReduction
LIDCArtifactReduction.init(gpu_id=0)
#LIDCArtifactReduction.init(gpu_memory_limit_MB=6700)

import numpy as np
import matplotlib.pyplot as plt

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.array_streams import RecSinoArrayStream
from LIDCArtifactReduction.directory_system import DirectorySystem
from LIDCArtifactReduction.generator.generator import LIDCDataGenerator
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNetTraining import IterativeARTResNetTraining
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet_generator_transform import \
    IterativeARTResNetGeneratorTransform
from LIDCArtifactReduction.radon_transformation.radon_transformation_pyronn import PyronnParallelARTRadonTransform
from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.data_formatter import input_data_decoder
from LIDCArtifactReduction.utility import show_grey


geometry = RadonGeometry(volume_img_width=256, projection_width=256, nr_projections=40)
radon_transform = PyronnParallelARTRadonTransform(geometry, alfa=0.5 / 256)

ds = DirectorySystem(geometry, data_name='pyronn', algorithm_name='IterativeARTResnet')
array_stream = RecSinoArrayStream(directory=ds.DATA_DIRECTORY)  # or ds.SAMPLE_DATA_DIRECTORY

generator = LIDCDataGenerator(array_stream=array_stream, data_configuration_dir=ds.DATA_CONFIGURATION_DIR,
                            load_data_config='config20201028-031556', verbose=True, batch_size=10)
# For noisy transformations you may set lnI0 and sumscaling.
noisy_transformer = IterativeARTResNetGeneratorTransform(geometry, radon_transform, add_noise=True,
                                                        mode=4, lnI0=10*np.log(5), sum_scaling=5.0)
test_iterator = generator.get_new_test_iterator(noisy_transformer)

resnet = IterativeARTResNet(radon_geometry=geometry, radon_transformation=radon_transform,
                            name='Sample_IterativeARTResNet')
training_network = IterativeARTResNetTraining(radon_transformation=radon_transform, dir_system=ds,
                                        target_model=resnet, name='Sample_IterativeARTResNet_Training')
# Load weights. latest=True loads the most recent weightfile and ignores 'name'.
training_network.load_weights(name='Sample_IterativeARTResNet_Training51.619')#latest=True)

test_batch = next(test_iterator)
actual_reconstructions, bad_sinograms, good_reconstructions = input_data_decoder(test_batch)


def plot_in_row(imgs, norm_values, title=None):
    nr_imgs = len(imgs)
    fig = plt.figure(figsize=(nr_imgs * 5, 5))
    for i in range(nrOfSamples):
        fig.add_subplot(1, nr_imgs, i+1)
        show_grey(imgs[i], norm_values=norm_values)
        if title:
            plt.title(title)
    plt.show()

def plot_reconstructions(recs, title=None):
    plot_in_row(recs, norm_values=[-1.0, 3.0], title=title)

def plot_sinograms(sinos, title=None):
    plot_in_row(sinos, norm_values=[-5.0, 300.], title=title)


nrOfSamples = 6

plot_reconstructions(good_reconstructions.numpy(), title='Ideal reconstruction')
plot_reconstructions(actual_reconstructions.numpy(), title='Init: 4 ART iters')

for i in range(5):
    inputs = {IterativeARTResNet.imgs_input_name: actual_reconstructions, IterativeARTResNet.sinos_input_name: bad_sinograms}
    actual_reconstructions, errors_sinogram = training_network(inputs)
    plot_reconstructions(actual_reconstructions.numpy(), title='Init + {}x(ART+ResNet)'.format(i+1))
