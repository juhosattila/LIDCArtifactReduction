import platform

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction import utility


if platform.system() == 'Windows':
    PROJECT_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction"
    PROJECT_DATA_DIRECTORY = utility.direc(PROJECT_DIRECTORY, 'LIDCArtifactReduction_Data')

else:  # 'Linux'
    PROJECT_DIRECTORY = '/home/juhosa/LIDCArtifactReduction/'
    PROJECT_DATA_DIRECTORY = '/home/juhosa/CI/LIDCArtifactReduction_Data'


MODEL_PLOTS_DIRECTORY = utility.direc(PROJECT_DIRECTORY, 'model_plots')
#MODEL_DIRECTORY = utility.direc(PROJECT_DATA_DIRECTORY, 'models')
MODEL_WEIGHTS_DIRECTORY = utility.direc(PROJECT_DATA_DIRECTORY, 'model_weights')

DATA_CONFIGURATION_DIR = utility.direc(PROJECT_DATA_DIRECTORY, 'data_config')
PREDICTED_IMAGES_DIR = utility.direc(PROJECT_DATA_DIRECTORY, 'predictions')


class DirectorySystem:
    def __init__(self, geometry: RadonGeometry, data_name: str = '', algorithm_name: str = ''):
        """
        Args:
            data_name: suffix added to data directories name
            algorithm_name: suffix added to concrete system's directory name
        """
        if platform.system() == 'Windows':
            self.PROJECT_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction"
            self.PROJECT_DATA_DIRECTORY = utility.direc(PROJECT_DIRECTORY, 'LIDCArtifactReduction_Data')

        else:  # 'Linux'
            self.PROJECT_DIRECTORY = '/home/juhosa/LIDCArtifactReduction/'
            self.PROJECT_DATA_DIRECTORY = '/home/juhosa/CI/LIDCArtifactReduction_Data'

        data_prefix = '-' + data_name
        configuration = data_prefix + '-a' + str(geometry.nr_projections)
        self.DATA_DIRECTORY = utility.direc(PROJECT_DATA_DIRECTORY, 'LIDC-IDRI-transformed' + configuration)
        self.SAMPLE_DATA_DIRECTORY = utility.direc(PROJECT_DATA_DIRECTORY,
                                                   'LIDC-IDRI-sample-transformed' + configuration)


        algorithm_dir = 'algorithm-' + algorithm_name
        self.ALGORITHM_DIRECTORY = utility.direc(PROJECT_DATA_DIRECTORY, algorithm_dir)
        self.LOG_DIRECTORY = utility.direc(self.ALGORITHM_DIRECTORY, 'logs')
        self.TENSORBOARD_LOGDIR = utility.direc(self.LOG_DIRECTORY, 'tensorboard')
        self.CSV_LOGDIR = utility.direc(self.LOG_DIRECTORY, 'csvlogger')
