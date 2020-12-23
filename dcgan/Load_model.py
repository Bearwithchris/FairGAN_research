# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:20:06 2020
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://towardsdatascience.com/dcgans-generating-dog-images-with-tensorflow-and-keras-fb51a1071432
@author: Chris
"""

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf
import data
from functools import partial
import model_exp as model
from IPython import display

import config as c

image_size = 128
batch_size = 128
NOISE_DIM = 100
# noiseratio=1

c.config_gpu()

#######Experimental Function######################################
def smooth_positive_labels(y):
    return y - 0.3 + (tf.random.uniform(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + tf.random.uniform(y.shape) * 0.3

def flip(x: tf.Tensor) -> (tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    return x
##################################################################


generator_optimizer = tf.keras.optimizers.Adam(0.0002,beta_1=0.5 )
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002,beta_1=0.5 )


def discriminator_loss(real_output, fake_output):
    cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    real_loss = cross_entropy(tf.ones_like(real_output)*0.9, real_output) #Experiment One side smoothening 
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # real_loss = cross_entropy(smooth_positive_labels(tf.ones_like(real_output)), real_output) #Experiment
    # fake_loss = cross_entropy(smooth_negative_labels(tf.zeros_like(fake_output)), fake_output) #Experiment
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
   

discriminator = model.make_discriminator_model_128()
generator = model.make_generator_model_128()



def load_model(checkpoint_dir):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return generator


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5 )
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  

def generate_images(path,num_images,prefix):
    batch_size=9
    def g_and_s(model, epoch, test_input,prefix):
      # Notice `training` is set to False.
      # This is so all layers run in inference mode (batchnorm).
      predictions = model(test_input, training=False)
    
      fig = plt.figure(figsize=(3,3))
    
      for i in range(predictions.shape[0]):
          plt.subplot(3, 3, i+1)
          plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5 )
          plt.axis('off')
    
      plt.savefig(str(prefix)+'_{:04d}.png'.format(epoch))
      plt.show()
      
    noise=tf.random.normal([num_images,batch_size,100])
    model=load_model(path)
    for i in range(num_images):
        g_and_s(model,i,noise[i],prefix)


# generate_images("./training_checkpoints_unweighted_biasPoint9_2212",14,"unweighted_bias_point9")
# generate_images("./training_checkpoints_weighted_biasPoint9_2212",14,"weighted_bias_point9")
# generate_images("./training_checkpoints_run3_best_unbias",14,"unbias")