# For testing.

from tensorflow.python.framework import ops
import pyronn_layers
import tensorflow as tf


# parallel_projection2d
#@tf.function
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
            volume_origin=tf.cast(tf.broadcast_to(geometry.volume_origin, tf.concat([[batch], tf.shape(geometry.volume_origin)], axis=0)), dtype=tf.float32),
            detector_origin=tf.cast(tf.broadcast_to(geometry.detector_origin, tf.concat([[batch], tf.shape(geometry.detector_origin)], axis=0)), dtype=tf.float32),
            volume_spacing=tf.cast(tf.broadcast_to(geometry.volume_spacing, tf.concat([[batch], tf.shape(geometry.volume_spacing)], axis=0)), dtype=tf.float32),
            detector_spacing=tf.cast(tf.broadcast_to(geometry.detector_spacing, tf.concat([[batch], tf.shape(geometry.detector_spacing)], axis=0)), dtype=tf.float32),
            ray_vectors=tf.cast(tf.broadcast_to(geometry.ray_vectors, tf.concat([[batch], tf.shape(geometry.ray_vectors)], axis=0)), dtype=tf.float32))


# parallel_backprojection2d
#@tf.function
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
            volume_origin   =tf.cast(tf.broadcast_to(geometry.volume_origin,tf.concat([[batch],tf.shape(geometry.volume_origin)], axis=0)), dtype=tf.float32),
            detector_origin =tf.cast(tf.broadcast_to(geometry.detector_origin,tf.concat([[batch],tf.shape(geometry.detector_origin)], axis=0)), dtype=tf.float32),
            volume_spacing  =tf.cast(tf.broadcast_to(geometry.volume_spacing,tf.concat([[batch],tf.shape(geometry.volume_spacing)], axis=0)), dtype=tf.float32),
            detector_spacing=tf.cast(tf.broadcast_to(geometry.detector_spacing,tf.concat([[batch],tf.shape(geometry.detector_spacing)], axis=0)), dtype=tf.float32),
            ray_vectors     =tf.cast(tf.broadcast_to(geometry.ray_vectors,tf.concat([[batch],tf.shape(geometry.ray_vectors)], axis=0)), dtype=tf.float32))



