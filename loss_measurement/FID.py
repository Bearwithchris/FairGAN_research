# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:39:00 2020

@author: Chris
"""

# example of calculating the frechet inception distance in Keras for cifar10
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
# from keras.datasets import cifar10

import generator as g
import tensorflow as tf
from functools import partial
import data_tensorflow as datat


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


batch_size=10000
unbias_dir = 'D:/GIT/local_data_in_use/unbias'
dim=128

#Inception Model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(dim,dim,3))

#Unbias Image Preprocessing
unbias_list = tf.data.Dataset.list_files(unbias_dir + '/*')        
preprocess_function = partial(datat.preprocess_image, target_size=dim)  #Partially fill in a function data.preprocess_image with the arguement image_size
unbias_data = unbias_list.map(preprocess_function).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds
images1=iter(unbias_data).get_next().numpy()

#Generator model
gen_unweighted=g.generator("unweighted_model",dim)
images2=gen_unweighted.generate(batch_size)

# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

# calculate fid
fid = calculate_fid(model, images1, images2)
print('Unweighted FID: %.3f' % fid)


#Generator model (Weighted)
gen_weighted=g.generator("weighted_model",dim)
images2=gen_unweighted.generate(batch_size)
images2 = preprocess_input(images2)


# calculate fid
fid = calculate_fid(model, images1, images2)
print('Weighted FID: %.3f' % fid)