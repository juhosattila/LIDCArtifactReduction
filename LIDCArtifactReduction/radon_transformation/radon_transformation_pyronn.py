import numpy as np

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike

# from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
# from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from LIDCArtifactReduction.radon_transformation.pyronn_wrapper import *
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters import filters

from LIDCArtifactReduction.radon_transformation.radon_geometry import RadonGeometry
from LIDCArtifactReduction.radon_transformation.radon_transformation_abstracts import ForwardprojectionRadonTransform, \
    BackprojectionRadonTransform, RadonTransform, ARTRadonTransform


def get_pyronn_geometry(radon_geometry: RadonGeometry):
    # Volume Parameters:
    volume_size = radon_geometry.volume_img_width
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = radon_geometry.projection_width
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = radon_geometry.nr_projections
    angular_range = np.radians(180.)

    # create Geometry class
    pyronn_geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing,
                                  number_of_projections, angular_range)
    pyronn_geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(pyronn_geometry))
    return pyronn_geometry


class PyronnParallelForwardprojectionRadonTransform(ForwardprojectionRadonTransform):
    def __init__(self, radon_geometry: RadonGeometry):
        self._geometry = get_pyronn_geometry(radon_geometry)

    # Always convert arguments to tensor before passing and make sure, that shape is constant,
    # otherwise new and new graphs will be created, slowing down the execution.
    # In this case size of tensor is varying, hence it is slow. Do not use yet.
    # @tf.function
    def forwardproject(self, imgs: TensorLike):
        imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
        sinos_3D = parallel_projection2d(imgs_tf, self._geometry)  # removes channeldimension
        sinos_4D = tf.expand_dims(sinos_3D, axis=-1)
        return sinos_4D


class PyronnParallelBackprojectionRadonTransform(BackprojectionRadonTransform):
    def __init__(self, radon_geometry: RadonGeometry):
        self._geometry = get_pyronn_geometry(radon_geometry)

    #@tf.function
    def backproject(self, sinos: TensorLike):
        sinos_tf = tf.convert_to_tensor(sinos, dtype=tf.float32)
        recos_3D = parallel_backprojection2d(sinos_tf, self._geometry)  # removes channeldimension
        recos_4D = tf.expand_dims(recos_3D, axis=-1)
        return recos_4D


class PyronnParallelRadonTransform(RadonTransform):
    def __init__(self, radon_geometry: RadonGeometry):
        self._geometry = get_pyronn_geometry(radon_geometry)

    #@tf.function
    def forwardproject(self, imgs: TensorLike):
        imgs_tf = tf.convert_to_tensor(imgs, dtype=tf.float32)
        sinos_3D = parallel_projection2d(imgs_tf, self._geometry)  # removes channeldimension
        sinos_4D = tf.expand_dims(sinos_3D, axis=-1)
        return sinos_4D

    #@tf.function
    def backproject(self, sinos: TensorLike):
        sinos_tf = tf.convert_to_tensor(sinos, dtype=tf.float32)
        recos_3D = parallel_backprojection2d(sinos_tf, self._geometry)  # removes channeldimension
        recos_4D = tf.expand_dims(recos_3D, axis=-1)
        return recos_4D

    #@tf.function
    def invert(self, sinos: TensorLike):
        raise NotImplementedError()

        # Converting back to real is smelly.

        # sinos_tf_34D = tf.convert_to_tensor(sinos, dtype=tf.float32)
        # sinos_tf_3D = tf.reshape(sinos_tf_34D, shape=tf.shape(sinos_tf_34D)[:3])
        #
        # reco_filter = filters.ram_lak_2D(self._geometry)
        # sino_freq = tf.signal.fft(tf.cast(sinos_tf_3D, dtype=tf.complex64))
        # sino_filtered_freq = tf.multiply(sino_freq, tf.cast(reco_filter, dtype=tf.complex64))
        # sinogram_filtered = tf.math.real(tf.signal.ifft(sino_filtered_freq))
        #
        # reco = parallel_backprojection2d(sinogram_filtered, geometry)


class PyronnParallelARTRadonTransform(PyronnParallelRadonTransform, ARTRadonTransform):
    def __init__(self, radon_geometry: RadonGeometry, alfa):
        PyronnParallelRadonTransform.__init__(self, radon_geometry=radon_geometry)
        ARTRadonTransform.__init__(self, tf.convert_to_tensor(alfa, dtype=tf.float32))

    def ART_step(self, imgs: TensorLike, sinos: TensorLike):
        return imgs + self.alfa * self.backproject(sinos - self.forwardproject(imgs))
