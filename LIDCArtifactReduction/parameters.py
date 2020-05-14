import platform, os
from LIDCArtifactReduction.utility import direc

# Radon geometry
IMG_SIDE_LENGTH = 256
NR_OF_SPARSE_ANGLES = 40

# HU scaling
HU_TO_CT_SCALING = 1000.0

if platform.system() == 'Windows':
    PROJECT_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction"
    PROJECT_DATA_DIRECTORY = direc(PROJECT_DIRECTORY, 'LIDCArtifactReduction_Data')

else:  # 'Linux'
    PROJECT_DIRECTORY = '/home/juhosa/LIDCArtifactReduction/'
    PROJECT_DATA_DIRECTORY = '/home/juhosa/CI/LIDCArtifactReduction_Data'

DATA_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'LIDC-IDRI-transformed-40')
TEST_DATA_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'LIDC-IDRI-sample-transformed-40')
MODEL_PLOTS_DIRECTORY = direc(PROJECT_DIRECTORY, 'model_plots')
MODEL_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'models')
MODEL_WEIGHTS_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'model_weights')

LOG_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'logs')
TENSORBOARD_LOGDIR = direc(LOG_DIRECTORY, 'tensorboard')
CSV_LOGDIR = direc(LOG_DIRECTORY, 'csvlogger')
DATA_CONFIGURATION_DIR = direc(PROJECT_DATA_DIRECTORY, 'data_config')

PREDICTED_IMAGES_DIR = direc(PROJECT_DATA_DIRECTORY, 'predictions')
