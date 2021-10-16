from imageio import imread, imsave
from skimage.util import img_as_ubyte
import os
import numpy as np


def img_read(path, is_label=False):
	im = imread(path)
	if not is_label and im.shape[2] > 3:
		im = im[:, :, :3]
	return im


def img_save(im, path):
	directory = os.path.dirname(path)
	os.makedirs(directory, exist_ok=True)
	imsave(path, img_as_ubyte(im))


def label_save(label, path, num_of_labels=50):
	label_colours = np.random.randint(255, size=(num_of_labels, 3))
	inds_rgb = np.array([label_colours[c % num_of_labels] for c in label.astype( np.uint8 )])
	inds_rgb = inds_rgb.reshape(label.shape[:-1] + (3,))
	imsave(path, img_as_ubyte(inds_rgb))
