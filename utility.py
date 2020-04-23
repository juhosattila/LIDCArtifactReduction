import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def show_grey(img_or_imgs):
    temp = np.asarray(img_or_imgs)
    shape = np.shape(temp)
    if shape[-1] == 1:
        temp = np.reshape(temp, newshape=shape[:-1])

    if np.ndim(temp) == 2:
        temp = [temp]

    for img in temp:
        plt.imshow(img, cmap=plt.cm.Greys_r)
        plt.show()