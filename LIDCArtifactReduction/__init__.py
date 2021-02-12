def init(gpu_id=0, gpu_memory_limit_MB=None):
    """Initialising the package is mandatory. Initialises GPU id, GPU memory and the Tensorflow Addons library.

    The Tensorflow Addons library must be initialised in order to be able to save and load Radon layers.

    Args:
        gpu_id: should be id of GPU in the list produced by tf.config.experimental.list_physical_devices('GPU').
            NOW:
                0 -- Titan Xp
                1 -- GTX 1080 Ti
        gpu_memory_limit_MB: could be None, in which case the system allocates the amount of memory necessary.
    """
    import tensorflow as tf
    import platform

    if platform.system() == 'Linux':
        # Add my package folder, if necessary.
        # sys.path.insert(1, "/home/juhosa/python-packages")
        # print(sys.path)

        physical_gpus = tf.config.list_physical_devices('GPU')
        used_physical_gpu = physical_gpus[gpu_id]
        tf.config.set_visible_devices([used_physical_gpu], 'GPU')
        try:
            if gpu_memory_limit_MB is not None:
                tf.config.set_logical_device_configuration(used_physical_gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_MB)])
                print("Tensorflow: memory limit is set to {}".format(gpu_memory_limit_MB))

            # # TODO: somwhow solve this issue with PyPi pyronn package.
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            # print("Tensorflow: memory_growth is set to True")

            logical_gpus = tf.config.list_logical_devices('GPU')
            logical_cpus = tf.config.list_logical_devices('CPU')
        except RuntimeError as e:
            print(e)


    # Register in order to be able to save and load Radon layers.
    import tensorflow_addons as tfa
    from packaging import version
    if version.parse(tfa.__version__) >= version.parse("0.9.1"):
        tfa.register_all()
