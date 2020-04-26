import numpy as np
from dicom_preprocess import DicomLoader
from tf_image import ParallelRadonTransform, scale_HU2Radio
import tensorflow as tf
import parameters


#@tf.function
def run_transformations(data, radon_trans):
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    data_tf = tf.expand_dims(data_tf, axis=-1)
    resized_data = scale_HU2Radio(tf.image.resize(data_tf, size=[256, 256]))
    data_sino = radon_trans(resized_data)
    return resized_data.numpy(), data_sino.numpy()

def run():
    dl = DicomLoader(batch_size=5)
    radon_trans = ParallelRadonTransform(img_side_length=parameters.IMG_SIDE_LENGTH,
                                         angles=np.linspace(0.0, 180.0, parameters.NR_OF_SPARSE_ANGLES))

    for data_batch in dl:
        data_resized, data_sino = run_transformations(data_batch, radon_trans)
    # TODO: continue



if __name__ == "__main__":
    run()


