import LIDCArtifactReduction
#LIDCArtifactReduction.init(gpu_memory_limit_MB=6700)
LIDCArtifactReduction.init()

from LIDCArtifactReduction import parameters
#parameters.create(number_of_angles_parallel_radon=40)

import numpy as np
from datetime import datetime
from LIDCArtifactReduction.radon_params import RadonParams
from LIDCArtifactReduction.neural_nets.DCAR_UNet_FewBatchNorms import DCAR_UNet_FewBatchNorms
from LIDCArtifactReduction.neural_nets.DCAR_TrainingNetwork import DCAR_TrainingNetwork
from LIDCArtifactReduction.generator import LIDCDataGenerator
from LIDCArtifactReduction.array_streams import RecSinoArrayStream

from LIDCArtifactReduction.utility import show_grey
from LIDCArtifactReduction import utility


radon_params = RadonParams(angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES),
                           projection_width=parameters.IMG_SIDE_LENGTH)

unet = DCAR_UNet_FewBatchNorms(name='UNet_L1TV_ValRec_Sample', conv_regularizer=0.5 * 1e-3)
training_network = DCAR_TrainingNetwork(radon_params, target_model=unet,
                                        name='UNet_L1TV_ValRec_Sample_Training')

training_network.load_weights(name='UNet_Transferred_NoTV_Training.06-0.0042')#latest=True)

generator = LIDCDataGenerator(verbose=True, batch_size=5, load_data_config=True, shuffle=True,
                              array_stream=RecSinoArrayStream(directory=parameters.TEST_DATA_DIRECTORY))
test_batch = next(generator.test_iterator)
test_batch = test_batch[:2]
results = training_network(test_batch)
results = (results[0].numpy(), results[1].numpy())

nr_of_samples = 3
timestemp = datetime.now().strftime("%Y%m%d-%H%M%S")
base_dir = utility.direc(parameters.PREDICTED_IMAGES_DIR, 'UNet_Transferred_NoTV_Training.06-0.0042' )

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

