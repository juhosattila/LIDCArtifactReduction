import platform, os

# Radon geometry
IMG_SIDE_LENGTH = 256
NR_OF_SPARSE_ANGLES = 60

# HU scaling
HU_TO_CT_SCALING = 1000

if platform.system() == 'Windows':
    PROJECT_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction"
else:  # 'Linux' # TODO: rethink it, should go to hard disk
    PROJECT_DIRECTORY = '~/LIDCArtifactReduction/'

DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'images')
TEST_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'sample_images')

MODEL_PLOTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'model_plots')
MODEL_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'models')
MODEL_WEIGHTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'model_weights')


# -------------------------------------------------------#
if not os.path.exists(MODEL_PLOTS_DIRECTORY):
    os.mkdir(MODEL_PLOTS_DIRECTORY)
if not os.path.exists(MODEL_DIRECTORY):
    os.mkdir(MODEL_DIRECTORY)
if not os.path.exists(TEST_DIRECTORY):
    os.mkdir(TEST_DIRECTORY)
if not os.path.exists(MODEL_WEIGHTS_DIRECTORY):
    os.mkdir(MODEL_WEIGHTS_DIRECTORY)
