# Radon geometry
IMG_SIDE_LENGTH = 256
NR_OF_SPARSE_ANGLES = 60

# HU scaling
HU_TO_CT_SCALING = 1000

import platform, os
if platform.system() == 'Windows':
    PROJECT_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction"
else: # 'Linux'
    PROJECT_DIRECTORY = '~/LIDCArtifactReduction/'

DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'images')
TEST_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'sample_images')