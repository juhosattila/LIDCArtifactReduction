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

generator = LIDCDataGenerator(verbose=True)
training_network.fit(generator.train_iterator, generator.valid_iterator,
                     epochs=5, adam_lr=1e-3)
