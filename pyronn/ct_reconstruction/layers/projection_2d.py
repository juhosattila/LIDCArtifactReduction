# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.python.framework import ops
import pyronn_layers
import numpy as np
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
    batch = np.shape(volume)[0]
    return pyronn_layers.parallel_projection2d(volume,
                                               projection_shape=geometry.sinogram_shape,
                                               volume_origin=np.broadcast_to(geometry.volume_origin, [batch, *np.shape(geometry.volume_origin)]),
                                               detector_origin=np.broadcast_to(geometry.detector_origin, [batch, *np.shape(geometry.detector_origin)]),
                                               volume_spacing=np.broadcast_to(geometry.volume_spacing, [batch, *np.shape(geometry.volume_spacing)]),
                                               detector_spacing=np.broadcast_to(geometry.detector_spacing, [batch, *np.shape(geometry.detector_spacing)]),
                                               ray_vectors=np.broadcast_to(geometry.ray_vectors, [batch, *np.shape(geometry.ray_vectors)]))


