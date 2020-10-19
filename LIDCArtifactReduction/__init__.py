def init(gpu_memory_limit_MB=None):
    """Initialising the package is mandatory. Initialises GPU memory and the Tensorflow Addons library.

    The Tensorflow Addons library must be initialised in order to be able to save and load Radon layers.

    Args:
        gpu_memory_limit_MB: could be None, in which case the system allocates the amount of memory necessary.
    """
    import tensorflow as tf
    import platform

    def set_memory_limit(gpu_mem_limit):
        # Purely dependent of server nhakni.
        if platform.system() == 'Linux':
            # Add my package folder, if necessary.
            # sys.path.insert(1, "/home/juhosa/python-packages")
            # print(sys.path)

            # Set resource limit of tensorflow on server
            gpus = tf.config.experimental.list_physical_devices('GPU')
            try:
                # Only set for gpus[0].
                tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])
                print("Tensorflow: memory limit is set to {}".format(gpu_mem_limit))

                # # TODO: somwhow solve this issue with PyPi pyronn package.
                # tf.config.experimental.set_memory_growth(gpus[0], True)
                # print("Tensorflow: memory_growth is set to True")

                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)


    # Set GPU memory limit if necessary
    if gpu_memory_limit_MB is not None:
        set_memory_limit(gpu_mem_limit=gpu_memory_limit_MB)


    # Register in order to be able to save and load Radon layers.
    import tensorflow_addons as tfa
    from packaging import version
    if version.parse(tfa.__version__) >= version.parse("0.9.1"):
        tfa.register_all()
