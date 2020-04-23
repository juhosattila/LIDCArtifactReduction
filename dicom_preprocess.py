import pylidc as pl
import numpy as np
from skimage.exposure import rescale_intensity
import os
from utility import show_grey


FILEDIR = 'images'
FILENAME = 'sample_imgs.npz'
IMG_ARRAY_NAME= "imgs"


file = os.path.join(FILEDIR, FILENAME)

def save_imgs(imgs):
    np.savez_compressed(file, **{IMG_ARRAY_NAME :imgs})

def run():
    pid = 'LIDC-IDRI-0001'
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = np.transpose(scan.to_volume(), [2, 0, 1])
    #one_pic = vol[55]
    multiple_pics = vol[32:64]

    #pic_rescaled = rescale_intensity(one_pic, out_range=(0.0, 1.0))
    imgs_rescaled = rescale_intensity(multiple_pics, out_range=(0.0, 1.0))
    save_imgs(imgs_rescaled)

def load_sample_imgs():
    with np.load(file) as data:
        sample_img = data[IMG_ARRAY_NAME]
    return sample_img

def test_loading():
    imgs = load_sample_imgs()
    print(np.shape(imgs))
    show_grey(imgs[:5])

if __name__ == "__main__":
    run()
    test_loading()