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
image_size = 64
batch_size = 128
NOISE_DIM = 100
# noiseratio=1


#######Experimental Function######################################
def smooth_positive_labels(y):
    return y - 0.3 + (tf.random.uniform(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + tf.random.uniform(y.shape) * 0.3

def flip(x: tf.Tensor) -> (tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    return x
##################################################################

# DATA_BASE_DIR="D:/GIT/local_data_in_use/dummy"
DATA_BASE_DIR="D:/GIT/local_data_in_use/bias_point9"
list_ds = tf.data.Dataset.list_files(DATA_BASE_DIR + '/*')    
preprocess_function = partial(data.preprocess_image, target_size=image_size)  #Partially fill in a function data.preprocess_image with the arguement image_size
# train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds
train_data = list_ds.map(preprocess_function).map(flip).shuffle(100).batch(batch_size)  #Experimental (Data Augmentation)


generator_optimizer = tf.keras.optimizers.Adam(0.0002,beta_1=0.5 )
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002,beta_1=0.5 )


def discriminator_loss(real_output, fake_output):
    cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # real_loss = cross_entropy(smooth_positive_labels(tf.ones_like(real_output)), real_output) #Experiment
    # fake_loss = cross_entropy(smooth_negative_labels(tf.zeros_like(fake_output)), fake_output) #Experiment
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
   
@tf.function
def train_step(generator, discriminator, real_image, batch_size):

    # noise = tf.random.normal([batch_size, noiseratio,noiseratio,NOISE_DIM])
    # noise = tf.random.uniform([batch_size,NOISE_DIM])
    noise = tf.random.normal([batch_size,NOISE_DIM])
    ###################################
    # Train D
    ###################################
    with tf.GradientTape() as g_tape,  tf.GradientTape(persistent=True) as d_tape:
        fake_image = generator(noise, training=True)
        
        real_output = discriminator(real_image, training=True)
        fake_output = discriminator(fake_image, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    # Calculate the gradients for discriminator
    g_gradients = g_tape.gradient(gen_loss, generator.trainable_variables)
    D_gradients = d_tape.gradient(disc_loss,discriminator.trainable_variables)
    
    # Apply the gradients to the optimizer
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(D_gradients,discriminator.trainable_variables))


discriminator = model.make_discriminator_model()
generator = model.make_generator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
SAVE_EVERY_N_EPOCH=10
MODEL_PATH = 'models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
def train(dataset, epochs):
  for epoch in range(epochs):
    print (epoch)
    start = time.time()

    for step, (image) in enumerate(dataset):
      # print(step)
      current_batch_size = image.shape[0]
      train_step(generator,discriminator,image,batch_size=tf.constant(current_batch_size, dtype=tf.int64))
      if step%100==0:
          generate_and_save_images(generator,epoch,tf.random.normal([16,NOISE_DIM]))
          
    if (epoch) % SAVE_EVERY_N_EPOCH == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    # Save the model every 15 epochs
    # if epoch % SAVE_EVERY_N_EPOCH == 0:
    #     generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_size)))
    #     discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_size)))
    #     print ('Saving model for epoch {}'.format(epoch))

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    

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
  
EPOCHS=20
train(train_data, EPOCHS)