import platform, os
from LIDCArtifactReduction.utility import direc

# Radon geometry
IMG_SIDE_LENGTH = 256
NR_OF_SPARSE_ANGLES = 60

# HU scaling
HU_TO_CT_SCALING = 1000.0

# DIRECTORIES
PROJECT_DIRECTORY = ''
PROJECT_DATA_DIRECTORY = ''
DATA_DIRECTORY = ''
TEST_DATA_DIRECTORY = ''
MODEL_PLOTS_DIRECTORY = ''
MODEL_DIRECTORY = ''
MODEL_WEIGHTS_DIRECTORY = ''
LOG_DIRECTORY = ''
TENSORBOARD_LOGDIR = ''
CSV_LOGDIR = ''
DATA_CONFIGURATION_DIR = ''
PREDICTED_IMAGES_DIR = ''


# Creation function for directories.
def create(image_side_length = None, number_of_angles_parallel_radon = None):
    # Need to specify variables as global in order to change them.
    global IMG_SIDE_LENGTH, NR_OF_SPARSE_ANGLES

    IMG_SIDE_LENGTH = image_side_length or IMG_SIDE_LENGTH
    NR_OF_SPARSE_ANGLES = number_of_angles_parallel_radon or NR_OF_SPARSE_ANGLES

    # Need to specify variables as global in order to change them.
    global PROJECT_DIRECTORY, PROJECT_DATA_DIRECTORY, DATA_DIRECTORY, TEST_DATA_DIRECTORY,\
        MODEL_PLOTS_DIRECTORY, MODEL_DIRECTORY, MODEL_WEIGHTS_DIRECTORY, LOG_DIRECTORY,\
        TENSORBOARD_LOGDIR, CSV_LOGDIR, DATA_CONFIGURATION_DIR, PREDICTED_IMAGES_DIR


    if platform.system() == 'Windows':
        PROJECT_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction"
        PROJECT_DATA_DIRECTORY = direc(PROJECT_DIRECTORY, 'LIDCArtifactReduction_Data')

    else:  # 'Linux'
        PROJECT_DIRECTORY = '/home/juhosa/LIDCArtifactReduction/'
        PROJECT_DATA_DIRECTORY = '/home/juhosa/CI/LIDCArtifactReduction_Data'

    angles_s = '-' + str(NR_OF_SPARSE_ANGLES)

    DATA_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'LIDC-IDRI-transformed' + angles_s)
    TEST_DATA_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'LIDC-IDRI-sample-transformed' + angles_s)
    MODEL_PLOTS_DIRECTORY = direc(PROJECT_DIRECTORY, 'model_plots')
    MODEL_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'models')
    MODEL_WEIGHTS_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'model_weights')

    LOG_DIRECTORY = direc(PROJECT_DATA_DIRECTORY, 'logs')
    TENSORBOARD_LOGDIR = direc(LOG_DIRECTORY, 'tensorboard')
    CSV_LOGDIR = direc(LOG_DIRECTORY, 'csvlogger')
    DATA_CONFIGURATION_DIR = direc(PROJECT_DATA_DIRECTORY, 'data_config')

    PREDICTED_IMAGES_DIR = direc(PROJECT_DATA_DIRECTORY, 'predictions')


# Initial create
create()
