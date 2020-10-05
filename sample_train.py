# Needed in this sequence
import LIDCArtifactReduction
#LIDCArtifactReduction.init()
LIDCArtifactReduction.init(gpu_memory_limit_MB=6700)

from LIDCArtifactReduction import parameters
#parameters.create(number_of_angles_parallel_radon=40)

import numpy as np
from LIDCArtifactReduction.radon_params import RadonParams
from LIDCArtifactReduction.neural_nets.DCAR_UNet_FewBatchNorms import DCAR_UNet_FewBatchNorms
from LIDCArtifactReduction.neural_nets.DCAR_TrainingNetwork import DCAR_TrainingNetwork
from LIDCArtifactReduction.generator import LIDCDataGenerator


radon_params = RadonParams(angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES),
                           projection_width=parameters.IMG_SIDE_LENGTH)

unet = DCAR_UNet_FewBatchNorms(name='UNet40', conv_regularizer=0.1 * 1e-3)
training_network = DCAR_TrainingNetwork(radon_params, target_model=unet, name='UNet40_Training')

# Optionally load weights. latest=True loads the most recent weightfile and ignores 'name'.
training_network.load_weights(name='UNet40_Training_weight_file')#latest=True)

# Setting training is always followed by compilation.
training_network.set_training(training=True)
training_network.compile(
    lr= 1e-5, reconstruction_output_weight=10.0, sino_output_weight=1.0 / (16.0 * parameters.NR_OF_SPARSE_ANGLES),
    add_total_variation=True, total_variation_eps=0.001, tot_var_loss_weight=0.05, mse_tv_weight=3.0)

generator = LIDCDataGenerator(load_data_config='data_config_file', verbose=True,
				validation_split=0.25, test_split=0.25, batch_size=8)
training_network.fit(generator.train_iterator, generator.valid_iterator, epochs=100,
                     steps_per_epoch=500, validation_steps=150,
					 verbose=1, initial_epoch=46, early_stoppig_patience=51)
