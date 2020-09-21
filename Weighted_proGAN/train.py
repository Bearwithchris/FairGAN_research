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
import density_classifier as dc

#Configure Gpus
c.config_gpu()
# c.precision() #WIP

image_size = 64

batch_size = 16
CURRENT_EPOCH = 189 # Epoch start from 1. If resume training, set this to the previous model saving epoch.

#Directories
# DATA_BASE_DIR="../../scratch/bias_point9"
# DATA_BASE_DIR2="../../scratch/unbias"
DATA_BASE_DIR="D:/GIT/local_data_in_use/bias_point9"
DATA_BASE_DIR2="D:/GIT/local_data_in_use/unbias"


#DATA_BASE_DIR="D:/GIT/ResearchCode/proGAN/alt"
TRAIN_LOGDIR = os.path.join("logs", "tensorflow", 'train_data') # Sets up a log directory.
MODEL_PATH = 'models'
OUTPUT_PATH = 'outputs'
PROGRESS='progress'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(PROGRESS):
    os.makedirs(PROGRESS)

#Progress log sections============================================================
#Create Progress Directory
def log_progress_active():
    if not os.path.isfile("./"+PROGRESS+"/progressLog.txt"):
        f = open("./"+PROGRESS+"/progressLog.txt", "w")
    else:
        f = open("./"+PROGRESS+"/progressLog.txt", "a")
    return f
        
def log_progress_deactive(f):
    f.close()
 
f=log_progress_active()
f.write("image_size= "+str(image_size)+" batch_size= "+str(batch_size)+" Current epoch= "+str(CURRENT_EPOCH)+"\n")
log_progress_deactive(f)
#=================================================================================

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



#Initiliase Data
list_ds = tf.data.Dataset.list_files(DATA_BASE_DIR + '/*')                    #Returns a tensor Dataset of file directory
preprocess_function = partial(data.preprocess_image, target_size=image_size)  #Partially fill in a function data.preprocess_image with the arguement image_size
train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds

list_ds2 = tf.data.Dataset.list_files(DATA_BASE_DIR2 + '/*')                    #Returns a tensor Dataset of file directory
preprocess_function2 = partial(data.preprocess_image, target_size=image_size)  #Partially fill in a function data.preprocess_image with the arguement image_size
train_data2 = list_ds2.map(preprocess_function2).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds

#Initilaise Model
generator, discriminator = m.model_builder(image_size)
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


#Define Optimiser
D_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)


def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, D_optimizer, G_optimizer):
    '''
        Set new learning rate to optimizersz
    '''
    K.set_value(D_optimizer.lr, new_lr)
    K.set_value(G_optimizer.lr, new_lr)
    
def calculate_batch_size(image_size):
    if image_size < 64:
        return 16
    elif image_size < 128:
        return 12
    elif image_size == 128:
        return 8
    elif image_size == 256:
        return 4
    else:
        return 3
    # if image_size <= 16:
    #     return 16
    # elif image_size <= 64:
    #     return 8
    # else:
    #     return 4
    
    
    
def generate_and_save_images(model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
    # Test input is a list include noise and label
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=figure_size)
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, '{}x{}_image_at_epoch_{:04d}.png'.format(predictions.shape[1], predictions.shape[2], epoch)))
    plt.show()

num_examples_to_generate = 9

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
# sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM], seed=0)
# sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)
# generate_and_save_images(generator, 0, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=False, is_flatten=False)


LAMBDA = 10


#==================TRAINING Functions======================================================================
def get_WGAN_GP_train_d_step_weighted():
    @tf.function
    def WGAN_GP_train_d_step_weighted(generator, discriminator, real_image, alpha, batch_size, step,weight):
        '''
            One training step
            
            Reference: https://www.tensorflow.org/tutorials/generative/dcgan
        '''
        w0,w1=tf.split(weight,num_or_size_splits=2,axis=1)
        weight_v=tf.math.divide(w0,w1)
        noise = tf.random.normal([batch_size, NOISE_DIM])
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        ###################################
        # Train D
        ###################################
        with tf.GradientTape(persistent=True) as d_tape:
            with tf.GradientTape() as gp_tape:
                fake_image = generator([noise, alpha], training=True)
                fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
                fake_mixed_pred = discriminator([fake_image_mixed, alpha], training=True)
                
            # Compute gradient penalty
            grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            # gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            gradient_penalty= tf.multiply(tf.reshape(weight_v,[batch_size]),tf.square(grad_norms - 1)) #weight addition for FairGAN
            
            fake_pred = discriminator([fake_image, alpha], training=True)
            real_pred = discriminator([real_image, alpha], training=True)
            
            # D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
            D_loss = tf.reduce_mean(tf.multiply(fake_pred,weight_v)) - tf.reduce_mean(tf.multiply(real_pred,weight_v)) + LAMBDA * gradient_penalty #weight addition for FairGAN
        # Calculate the gradients for discriminator
        D_gradients = d_tape.gradient(D_loss,discriminator.trainable_variables)
        # Apply the gradients to the optimizer
        D_optimizer.apply_gradients(zip(D_gradients,discriminator.trainable_variables))
        # Write loss values to tensorboard
        if step % 10 == 0:
            with file_writer.as_default():
                tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)
    return  WGAN_GP_train_d_step_weighted

def get_WGAN_GP_train_d_step():        
    @tf.function
    def WGAN_GP_train_d_step(generator, discriminator, real_image, alpha, batch_size, step):
        '''
            One training step
            
            Reference: https://www.tensorflow.org/tutorials/generative/dcgan
        '''
        noise = tf.random.normal([batch_size, NOISE_DIM])
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        ###################################
        # Train D
        ###################################
        with tf.GradientTape(persistent=True) as d_tape:
            with tf.GradientTape() as gp_tape:
                fake_image = generator([noise, alpha], training=True)
                fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
                fake_mixed_pred = discriminator([fake_image_mixed, alpha], training=True)
                
            # Compute gradient penalty
            grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            
            fake_pred = discriminator([fake_image, alpha], training=True)
            real_pred = discriminator([real_image, alpha], training=True)
            
            D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
        # Calculate the gradients for discriminator
        D_gradients = d_tape.gradient(D_loss,discriminator.trainable_variables)
        # Apply the gradients to the optimizer
        D_optimizer.apply_gradients(zip(D_gradients,discriminator.trainable_variables))
        # Write loss values to tensorboard
        if step % 10 == 0:
            with file_writer.as_default():
                tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)
    return WGAN_GP_train_d_step

def get_WGAN_GP_train_g_step():
    @tf.function
    def WGAN_GP_train_g_step(generator, discriminator, alpha, batch_size, step):
        '''
            One training step
            
            Reference: https://www.tensorflow.org/tutorials/generative/dcgan
        '''
        noise = tf.random.normal([batch_size, NOISE_DIM])
        ###################################
        # Train G
        ###################################
        with tf.GradientTape() as g_tape:
            fake_image = generator([noise, alpha], training=True)
            fake_pred = discriminator([fake_image, alpha], training=True)
            G_loss = -tf.reduce_mean(fake_pred)
        # Calculate the gradients for discriminator
        G_gradients = g_tape.gradient(G_loss,generator.trainable_variables)
        # Apply the gradients to the optimizer
        G_optimizer.apply_gradients(zip(G_gradients,generator.trainable_variables))
        # Write loss values to tensorboard
        if step % 10 == 0:
            with file_writer.as_default():
                tf.summary.scalar('G_loss', G_loss, step=step)
    return WGAN_GP_train_g_step



#Load old Models trained
# Load previous resolution model
# if image_size > 4:
#     if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_size / 2)))):
#         generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_size / 2))), by_name=True)
#         print("generator loaded")
#     if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_size / 2)))):
#         discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_size / 2))), by_name=True)
#         print("discriminator loaded")
        
        
# # To resume training, comment it if not using.
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size)))):
    generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size))), by_name=False)
    print("generator loaded")
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size)))):
    discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size))), by_name=False)
    print("discriminator loaded")

#load density classifier 
density_c=dc.generate_model(image_size,dc.classifier_best(image_size))

#===================================Actual Training======================================================================
total_data_number = len(os.listdir(DATA_BASE_DIR))
total_data_number2 = len(os.listdir(DATA_BASE_DIR2))
switch_res_every_n_epoch = 40


current_learning_rate = LR
training_steps = math.ceil(total_data_number / batch_size)
# Fade in half of switch_res_every_n_epoch epoch, and stablize another half
alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
alpha = min(1., (CURRENT_EPOCH - 1) % switch_res_every_n_epoch * training_steps *  alpha_increment)
EPOCHs = 320
SAVE_EVERY_N_EPOCH = 1 # Save checkpoint at every n epoch

sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM], seed=0)
sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)

#Initialise back propogation for @tf.function wrapper use
WGAN_GP_train_d_step_weighted=get_WGAN_GP_train_d_step_weighted()
WGAN_GP_train_d_step=get_WGAN_GP_train_d_step()
WGAN_GP_train_g_step=get_WGAN_GP_train_g_step()

for epoch in range(CURRENT_EPOCH, EPOCHs + 1):
    start = time.time()
    print('Start of epoch %d' % (epoch,))
    print('Current alpha: %f' % (alpha,))
    print('Current resolution: {} * {}'.format(image_size, image_size))
    
    f=log_progress_active()
    f.write("Start of epoch: %d     Current alpha: %f   Current resolution: %d x %d " % (epoch,alpha,image_size,image_size))
    f=log_progress_deactive(f)
    # Using learning rate decay
#     current_learning_rate = learning_rate_decay(current_learning_rate)
#     print('current_learning_rate %f' % (current_learning_rate,))
#     set_learning_rate(current_learning_rate) 
    
    #Biasdata
    alpha2=alpha
    for step, (image) in enumerate(train_data):
        current_batch_size = image.shape[0]
        alpha_tensor = tf.constant(np.repeat(alpha, current_batch_size).reshape(current_batch_size, 1), dtype=tf.float32)
        density_weight=density_c(image)
        # Train step
        WGAN_GP_train_d_step_weighted(generator, discriminator, image, alpha_tensor,
                              batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64),weight=density_weight)
        WGAN_GP_train_g_step(generator, discriminator, alpha_tensor,
                              batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
        
        
        # update alpha
        alpha = min(1., alpha + alpha_increment)
        
        #Loading screen
        if step % 10 == 0:
            print ('.', end='')
            
    #Unbias      
    for step2, (image2) in enumerate(train_data2):
        current_batch_size2 = image2.shape[0]
        alpha_tensor2 = tf.constant(np.repeat(alpha2, current_batch_size2).reshape(current_batch_size2, 1), dtype=tf.float32)
        density_weight=density_c(image)
        
        # Train step
        WGAN_GP_train_d_step(generator, discriminator, image2, alpha_tensor2,
                              batch_size=tf.constant(current_batch_size2, dtype=tf.int64), step=tf.constant(step2, dtype=tf.int64))
        WGAN_GP_train_g_step(generator, discriminator, alpha_tensor2,
                              batch_size=tf.constant(current_batch_size2, dtype=tf.int64), step=tf.constant(step2, dtype=tf.int64))
        
        
        # update alpha
        alpha2 = min(1., alpha2 + alpha_increment)
        
        #Loading screen
        if step % 10 == 0:
            print ('.', end='')
        
    
    # # Clear jupyter notebook cell output
    # clear_output(wait=True)
    
    # Using a consistent image (sample_X) so that the progress of the model is clearly visible.
    generate_and_save_images(generator, epoch, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)
    
    #Override old model and save over it
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_size)))
        discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_size)))
        print ('Saving model for epoch {}'.format(epoch))
    
    print ('Time taken for epoch {} is {} sec\n'.format(epoch,time.time()-start))
    f=log_progress_active()
    f.write("Time taken: %f \n" %(time.time()-start))
    log_progress_deactive(f)
    
    
    # Train next resolution
    if epoch % switch_res_every_n_epoch == 0:
        #Save weights one more time
        print('saving {} * {} model'.format(image_size, image_size))
        generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_size)))
        discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_size)))
        
        # Reset alpha
        alpha = 0
        
        #Save old image size
        previous_image_size = int(image_size)
        #Incerement to next image_size
        image_size = int(image_size * 2)
        
        if image_size > 512:
            print('Resolution reach 512x512, finish training')
            break
        
        #load density classifier 
        density_c=dc.generate_model(image_size,dc.classifier_best(image_size))

        #Load new model with increment dimension 
        print('creating {} * {} model'.format(image_size, image_size))
        generator, discriminator = m.model_builder(image_size)
        generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(previous_image_size, previous_image_size)), by_name=True)
        discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(previous_image_size, previous_image_size)), by_name=True)
        
        #Re-initilaise tf.graphs
        WGAN_GP_train_d_step_weighted=get_WGAN_GP_train_d_step_weighted()
        WGAN_GP_train_d_step=get_WGAN_GP_train_d_step()
        WGAN_GP_train_g_step=get_WGAN_GP_train_g_step()
        
        #Reprocessing new data in new size
        print('Making {} * {} dataset'.format(image_size, image_size))
        batch_size = calculate_batch_size(image_size)
        preprocess_function = partial(data.preprocess_image, target_size=image_size)
        train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)
        training_steps = math.ceil(total_data_number / batch_size)
        alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
        
        preprocess_function2 = partial(data.preprocess_image, target_size=image_size)
        train_data2 = list_ds2.map(preprocess_function2).shuffle(100).batch(batch_size)
        print('start training {} * {} model'.format(image_size, image_size))
        