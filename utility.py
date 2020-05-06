import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def show_grey(img_or_imgs):
    """
    :param img_or_imgs: should be an image (HWx) or batch of images (NHWx)
        or enumarble structure of images (e.g. list of tensors, images, that are HWx).
        They can be tensorflow or numpy images. If enumerable, then of arbitrary size.
        Images should not have channel information, only if the channel dimension has size 1.
        Hence x is either None or if C, than C=1.
        W must be more than 1, if there is no channel information.
    """
    if isinstance(img_or_imgs, tf.Tensor):
        img_or_imgs = img_or_imgs.numpy()

    if isinstance(img_or_imgs, np.ndarray):
        shape = np.shape(img_or_imgs)
        if shape[-1] == 1:
            img_or_imgs = np.reshape(img_or_imgs, newshape=shape[:-1])

        if np.ndim(img_or_imgs) == 2:
            img_or_imgs = [img_or_imgs]
    else:
        for i,_ in enumerate(img_or_imgs):
            if isinstance(img_or_imgs[i], tf.Tensor):
                img_or_imgs[i] = img_or_imgs[i].numpy()

            shape = np.shape(img_or_imgs[i])
            if shape[-1] == 1:
                img_or_imgs[i] = np.reshape(img_or_imgs[i], newshape=shape[:-1])

    for img in img_or_imgs:
        plt.imshow(img, cmap=plt.cm.Greys_r)
        #plt.imshow(img, cmap=plt.get_cmap('Greys'))
        plt.show()


def analyse(arrs, names=None):
    """Analyse list of numpy arrays in the form, of HW, NHW or NHWC
    """
    arrays = np.asanyarray(arrs, dtype='float32')
    if np.ndim(arrays) == 2:
        arrays = [arrays]

    def analyse_one(arr, name=None):
        print("-------------------------")
        print(f"Analysing array {name}:")
        print("Min: ", np.min(arr))
        print("Max: ", np.max(arr))
        print("-------------------------")

        show_grey(arr)
        plt.hist(arr)
        plt.show()

    if names is None:
        for arr in arrays:
            analyse_one(arr)

    else:
        if not isinstance(names, list):
            names = [names]
        for arr, name in zip(arrays, names):
            analyse_one(arr, name)
