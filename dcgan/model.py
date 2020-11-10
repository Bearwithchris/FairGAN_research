# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:00:16 2020

@author: Chris
"""

import tensorflow as tf

# def make_generator_model():
#     ngf=64
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.LeakyReLU())

#     model.add(tf.keras.layers.Reshape((7, 7, 256)))
#     # assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

#     model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     # assert model.output_shape == (None, 7, 7, 128)
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.LeakyReLU())

#     model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     # assert model.output_shape == (None, 14, 14, 64)
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.LeakyReLU())

#     model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#     # assert model.output_shape == (None, 28, 28, 1)

#     return model

# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[28, 28, 1]))
#     model.add(tf.keras.layers.LeakyReLU())
#     model.add(tf.keras.layers.Dropout(0.3))

#     model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(tf.keras.layers.LeakyReLU())
#     model.add(tf.keras.layers.Dropout(0.3))

#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(1))

#     return model

def make_generator_model():
    ngf=64
    inputs = tf.keras.layers.Input([1,1,100])
    
    #1
    x=tf.keras.layers.Conv2DTranspose(ngf*8,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #2
    x=tf.keras.layers.Conv2DTranspose(ngf*4,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #3
    x=tf.keras.layers.Conv2DTranspose(ngf*2,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #4
    x=tf.keras.layers.Conv2DTranspose(ngf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)

    #5
    x=tf.keras.layers.Conv2DTranspose(ngf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #6
    x=tf.keras.layers.Conv2DTranspose(3,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    out=tf.keras.activations.tanh(x)
    model=tf.keras.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

def make_discriminator_model():
    ndf=64
    inputs = tf.keras.layers.Input([64,64,3])
    
    #1
    x=tf.keras.layers.Conv2D(ndf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #2
    x=tf.keras.layers.Conv2D(ndf*2,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #3
    x=tf.keras.layers.Conv2D(ndf*4,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #4
    x=tf.keras.layers.Conv2D(ndf*8,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)

    # #5
    x=tf.keras.layers.Conv2D(ndf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    
    #6
    out=tf.keras.layers.Conv2D(1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    out=tf.keras.activations.sigmoid(out)
    model=tf.keras.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

# generator=make_generator_model()
# noise = tf.random.normal([1,1,1,100])
# generated_image = generator(noise, training=False)