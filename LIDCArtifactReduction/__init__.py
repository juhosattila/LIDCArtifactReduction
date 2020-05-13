def init(gpu_memory_limit_MB=None):
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
                tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)

    if gpu_memory_limit_MB is not None:
        set_memory_limit(gpu_mem_limit=gpu_memory_limit_MB)

    import tensorflow_addons as tfa
    # Register in order to be able to save and load Radon layers.
    from packaging import version
    if version.parse(tfa.__version__) >= version.parse("0.9.1"):
        tfa.register_all()
