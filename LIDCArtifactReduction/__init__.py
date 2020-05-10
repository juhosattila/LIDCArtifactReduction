import tensorflow as tf
# Set resource limit of tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
try:
     tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
except RuntimeError as e:
     print(e)

import tensorflow_addons as tfa
# Register in order to be able to save and load Radon layers.
tfa.register_all()