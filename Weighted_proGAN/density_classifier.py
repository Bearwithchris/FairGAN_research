# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:17:41 2020

@author: Chris
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:20:33 2020

@author: Chris
"""
import tensorflow as tf
import os
import fnmatch
import data
import numpy as np
import datetime as dt
from resnet import resnet18


# checkpoints="./training_checkpoints"
# checkpoints="./restnet18_bias_point9_training_checkpoints"




def create_model(dim):  
    model=resnet18.make_resnet_18(dim).call(isTrain=False)
    model.trainable=False
    
    optimizer=tf.keras.optimizers.Adam(1e-4)
    
    return model,optimizer

def load_checkpoints(directory):
    files = fnmatch.filter(os.listdir(directory), "*.index")
    files =[i.strip('.index') for i in files]
    return files
    
def restore_model(checkpoints,checkpoint_index,model,optimizer):
    checkpoint_dir = os.path.join(checkpoints, checkpoint_index)
    checkpoint = tf.train.Checkpoint(main_model=model, optimizer=optimizer)
    
    checkpoint.restore(checkpoint_dir)
    
    return model

def cross_entropy_loss(expected_labels, predicted_labels):
    cross_entropy =  tf.keras.losses.SparseCategoricalCrossentropy()
    loss = cross_entropy(expected_labels, predicted_labels)
    return loss 

def sort(listDir):
    numerics=[int(x.strip('ckpt-')) for x in listDir]
    placeHolder=["blank" for i in range(len(numerics))]
    for i in range(len(numerics)):
        placeHolder[numerics[i]-1]=listDir[i]
    return placeHolder

def classifier_best(dim):
    if dim==4:
        return 1
    elif dim==8:
        return 8
    elif dim==16:
        return 5
    elif dim==32:
        return 5
    elif dim==64:
        return 2
    elif dim==128:
        return 2
    else: 
        return 1
    

def generate_model(dim,model_epoch):
    # checkpoints="../DensityClassifier/training_checkpoints_"+str(dim)
    checkpoints="./Density_checkpoint/training_checkpoints_"+str(dim)
    listDir=load_checkpoints(checkpoints)
    listDir=sort(listDir)
    model,optimizer= create_model(dim)
    model=restore_model(checkpoints,listDir[model_epoch],model,optimizer)
    return model


def predict(model,image_batch):
    predicted_labels_test=model(image_batch)
    return predicted_labels_test
        
# dim=4
# test=generate_model(dim,classifier_best(dim))