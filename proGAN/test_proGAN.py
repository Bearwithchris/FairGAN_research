# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:11:15 2020

@author: Chris
"""
import tensorflow as tf 
import numpy as np
import model_tools as mt
import matplotlib.pyplot as plt
import model as m
from tensorflow.keras import backend as K
import os
from functools import partial
import data
import math
import time
from IPython.display import clear_output

import config as c

#Configure Gpus
c.config_gpu()
# c.precision() #WIP



batch_size = 16
CURRENT_EPOCH = 1 # Epoch start from 1. If resume training, set this to the previous model saving epoch.
image_size = 32

#Directories
# DATA_BASE_DIR="../../scratch/alt"
#DATA_BASE_DIR="D:/GIT/ResearchCode/proGAN/alt"
TRAIN_LOGDIR = os.path.join("logs", "tensorflow", 'train_data') # Sets up a log directory.
MODEL_PATH = 'models'
OUTPUT_PATH = 'Samples'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

# output_activation = tf.keras.activations.linear
output_activation = tf.keras.activations.tanh
kernel_initializer = 'he_normal'
NOISE_DIM = 512
LR = 1e-3
BETA_1 = 0.
BETA_2 = 0.99
EPSILON = 1e-8

# Decay learning rate
MIN_LR = 0.000001
DECAY_FACTOR=1.00004



#Initilaise Model
generator, discriminator = m.model_builder(image_size)
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)



    
    
def generate_and_save_images(model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
    # Test input is a list include noise and label
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=figure_size)
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, '{}x{}_samples_{:04d}.png'.format(predictions.shape[1], predictions.shape[2], epoch)))
    plt.show()

num_examples_to_generate = 9



LAMBDA = 10




        
# To resume training, comment it if not using.
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size)))):
    generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size))), by_name=False)
    print("generator loaded")
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size)))):
    discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size))), by_name=False)
    print("discriminator loaded")


#===================================Actual Training======================================================================
# total_data_number = len(os.listdir(DATA_BASE_DIR))
# switch_res_every_n_epoch = 40




samples=10

for i in range(samples):
    sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])
    sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)    
# Using a consistent image (sample_X) so that the progress of the model is clearly visible.
    generate_and_save_images(generator, CURRENT_EPOCH, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)
    CURRENT_EPOCH+=1    
