IMG_SIDE_LENGTH = 256
NR_OF_SPARSE_ANGLES = 60


import platform
if platform.system() == 'Windows':
    DATA_DIRECTORY = "C:\\Users\\juhos\\NemSzinkronizalt\\NN\\LIDCArtifactReduction\\images"
else: # 'Linux'
    # TODO :set
    raise NotImplementedError()
    DATA_DIRECTORY = '~/'