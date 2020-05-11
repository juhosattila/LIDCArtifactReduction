import numpy as np

import LIDCArtifactReduction
from LIDCArtifactReduction import parameters
from LIDCArtifactReduction.radon_params import RadonParams
from LIDCArtifactReduction.neural_nets.DCAR_UNet_FewBatchNorms import DCAR_UNet_FewBatchNorms
from LIDCArtifactReduction.neural_nets.DCAR_TrainingNetwork import DCAR_TrainingNetwork
from LIDCArtifactReduction.generator import LIDCDataGenerator


radon_params = RadonParams(angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES),
                           projection_width=parameters.IMG_SIDE_LENGTH)

unet = DCAR_UNet_FewBatchNorms(name='DCAR_UNet_FewBatchNorms')
training_network = DCAR_TrainingNetwork(radon_params, target_model=unet,
                                        name='DCAR_UNet_FewBatchNorms_Training')
training_network.set_training(training=True)
training_network.compile(
    adam_lr=1e-3, sino_output_weight=1.0 / parameters.NR_OF_SPARSE_ANGLES,
    total_variation_eps=1.0, tot_var_loss_weight=1e-3)

generator = LIDCDataGenerator(verbose=True, validation_split=0.25, test_split=0.25, batch_size=10)
training_network.fit(generator.train_iterator, generator.valid_iterator, epochs=5)
