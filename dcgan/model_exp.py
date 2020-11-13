# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:00:16 2020

@author: Chris
"""

import tensorflow as tf


weight_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)

def make_generator_model():
    ngf=64
    inputs = tf.keras.layers.Input(100)
    
    
    x=tf.keras.layers.Dense(4*4* ngf*8, use_bias=False)(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Reshape((4, 4, ngf*8))(x)
    # #1
    # x=tf.keras.layers.Conv2DTranspose(ngf*8,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(inputs)
    # x=tf.keras.layers.BatchNormalization()(x)
    # x=tf.keras.layers.ReLU()(x)
    
    #2 (4x4 -> 8x8)
    x=tf.keras.layers.Conv2DTranspose(ngf*4,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    
    
    #3 (8x8 -> 16x16)
    x=tf.keras.layers.Conv2DTranspose(ngf*2,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    
    #4 (16x16 -> 32x32)
    x=tf.keras.layers.Conv2DTranspose(ngf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)

    # #5
    # x=tf.keras.layers.Conv2DTranspose(ngf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    # x=tf.keras.layers.BatchNormalization()(x)
    # x=tf.keras.layers.ReLU()(x)
    
    #6 (32x32->64x64)
    x=tf.keras.layers.Conv2DTranspose(3,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    out=tf.keras.activations.tanh(x)
    
    model=tf.keras.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

# make_generator_model()

def make_discriminator_model():
    ndf=64
    inputs = tf.keras.layers.Input([64,64,3])

    
    #1 (64->32)
    x=tf.keras.layers.Conv2D(ndf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(inputs)
    # x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x=tf.keras.layers.Dropout(0.3)(x) #Test
    
    #2 (32->16)
    x=tf.keras.layers.Conv2D(ndf*2,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x=tf.keras.layers.Dropout(0.3)(x) #Test
    
    #3 (16->8)
    x=tf.keras.layers.Conv2D(ndf*4,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x=tf.keras.layers.Dropout(0.3)(x) #Test
    
    #4 (8->4)
    x=tf.keras.layers.Conv2D(ndf*8,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer=weight_initializer)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x=tf.keras.layers.Dropout(0.3)(x) #Test
    # # #5
    # x=tf.keras.layers.Conv2D(ndf*1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    # x=tf.keras.layers.BatchNormalization()(x)
    # x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    

    
    #6
    # out=tf.keras.layers.Conv2D(1,kernel_size=4,strides=2,padding='same',use_bias=False, kernel_initializer='glorot_normal')(x)
    
    out=tf.keras.layers.Flatten()(x)
    out=tf.keras.layers.Dense(1)(out)
    out=tf.keras.activations.sigmoid(out)
    
    model=tf.keras.Model(inputs=inputs, outputs=out)
    model.summary()
    return model

# make_discriminator_model()
# generator=make_generator_model()
# noise = tf.random.normal([1,1,1,100])
# generated_image = generator(noise, training=False)

