def init(gpu_id=0, gpu_memory_limit_MB : int=None):
    """Initialising the package is mandatory. Initialises GPU id, GPU memory.

    Args:
        gpu_id: should be id of GPU in the list produced by tf.config.experimental.list_physical_devices('GPU').
            NOW:
                0 -- Titan Xp
                1 -- GTX 1080 Ti
        gpu_memory_limit_MB: could be None, in which case the system allocates the amount of memory necessary.

    Example:
        init(gpu_id=0, gpu_memory_limit_MB=None)
        init(gpu_id=1, gpu_memory_limit_MB=5000)
    """
    import tensorflow as tf

    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    used_physical_gpu = physical_gpus[gpu_id]
    tf.config.set_visible_devices([used_physical_gpu], 'GPU')
    try:
        if gpu_memory_limit_MB is not None:

            tf.config.experimental.set_virtual_device_configuration(used_physical_gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit_MB)])
            print("Tensorflow: memory limit is set to {}".format(gpu_memory_limit_MB))
    except RuntimeError as e:
        print(e)

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
