from tensorflow.python.framework import ops
import pyronn_layers
import tensorflow as tf


# parallel_projection2d
# @tf.function
def parallel_projection2d(volume, geometry):
    """
    Wrapper function for making the layer call.
    Args:
        volume:     Input volume to project.
        geometry:   Corresponding GeometryParallel2D Object defining parameters.
    Returns:
            Initialized lme_custom_ops.parallel_projection2d layer.
    """
    batch = tf.shape(volume)[0]
    return pyronn_layers.parallel_projection2d(volume,
                                               projection_shape=geometry.sinogram_shape,
                                               volume_origin=tf.broadcast_to(geometry.volume_origin, [batch, *tf.shape(geometry.volume_origin)]),
                                               detector_origin=tf.broadcast_to(geometry.detector_origin, [batch, *tf.shape(geometry.detector_origin)]),
                                               volume_spacing=tf.broadcast_to(geometry.volume_spacing, [batch, *tf.shape(geometry.volume_spacing)]),
                                               detector_spacing=tf.broadcast_to(geometry.detector_spacing, [batch, *tf.shape(geometry.detector_spacing)]),
                                               ray_vectors=tf.broadcast_to(geometry.ray_vectors, [batch, *tf.shape(geometry.ray_vectors)]))


# parallel_backprojection2d
def parallel_backprojection2d(sinogram, geometry):
    """
    Wrapper function for making the layer call.
    Args:
        volume:     Input volume to project.
        geometry:   Corresponding GeometryParallel2D Object defining parameters.
    Returns:
            Initialized lme_custom_ops.parallel_backprojection2d layer.
    """
    batch = tf.shape(sinogram)[0]
    return pyronn_layers.parallel_backprojection2d(sinogram,
                                                   volume_shape=geometry.volume_shape,
                                                    volume_origin   =tf.broadcast_to(geometry.volume_origin,[batch,*tf.shape(geometry.volume_origin)]),
                                                    detector_origin =tf.broadcast_to(geometry.detector_origin,[batch,*tf.shape(geometry.detector_origin)]),
                                                    volume_spacing  =tf.broadcast_to(geometry.volume_spacing,[batch,*tf.shape(geometry.volume_spacing)]),
                                                    detector_spacing=tf.broadcast_to(geometry.detector_spacing,[batch,*tf.shape(geometry.detector_spacing)]),
                                                    ray_vectors     =tf.broadcast_to(geometry.ray_vectors,[batch,*tf.shape(geometry.ray_vectors)]))



