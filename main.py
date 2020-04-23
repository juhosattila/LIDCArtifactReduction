from RadonLayer import *
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tf_image import *
from utility import show_grey
import dicom_preprocess as dic
from skimage.transform import radon

from timeit import default_timer as timer


from DCAR_neural_nets import *


def RadonLayerTest():
    sample_indices = np.array([5, 10, 21])

    imgs_sample = dic.load_sample_imgs()
    show_grey(imgs_sample[sample_indices])

    start_time = timer()

    imgs_sample_tf = tf.constant(imgs_sample, dtype=tf.float32)

    print(f"Time for creating Tensors: {timer() - start_time}")
    start_time = timer()

    angles = np.linspace(0., 180., 60)
    radon_layer = RadonLayer(angles=angles)
    radon_layer.build(input_shape=tf.constant([1, 512, 512, 1]))

    print(f"Time for creating Radon Layer: {timer() - start_time}")
    start_time = timer()

    instances = imgs_sample_tf
    sinos = radon_layer.call(instances)

    print(f"Time for running layer: {timer() - start_time}")
    start_time = timer()

    print(f"Shape of TF sino: {tf.shape(sinos.numpy()[0])}")
    show_grey(sinos.numpy()[sample_indices])
    print(f"Shape of Numpy sino: {np.shape(radon(imgs_sample[0], angles))}")
    show_grey(radon(imgs_sample[0], angles))


def NetworkTest():
    radon_params = RadonParams(angles=np.linspace(0.0, 180.0, 60))
    unet = DCAR_UNet()
    training_network = DCAR_TrainingNetwork(radon_params, target_model=unet)

    #unet.summary()
    #training_network.summary()
    training_network.plot_model()

def GeneratorTest():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import Model





if __name__ == "__main__":
    #RadonLayerTest()
    NetworkTest()
    #GeneratorTest()
