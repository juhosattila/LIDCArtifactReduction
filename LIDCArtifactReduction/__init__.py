import sys
import tensorflow as tf
import platform
# Purely dependent of server nhakni.
if platform.system() == 'Linux':
    # Add my package folder, if necessary.
    # sys.path.insert(1, "/home/juhosa/python-packages")
    # print(sys.path)

    # Set resource limit of tensorflow on server
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)


import tensorflow_addons as tfa
# Register in order to be able to save and load Radon layers.
from packaging import version
if version.parse(tfa.__version__) >= version.parse("0.9.1"):
    tfa.register_all()
