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
from functools import partial
# import generator as g
import Load_model as l

import data_tensorflow as datat

#Resnet Checkpoint Path
checkpoints="./Resnet_training_Fairness_gender"
checkpoint_path="ckpt-7"

#Data Checkpoint Path
unbias_dir = 'D:/GIT/local_data_in_use/dref'


STORE_PATH="./"
# out_file_Test = STORE_PATH + f"/TEST_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
# train_summary_writer_Test = tf.summary.create_file_writer(out_file_Test)

dim=128
batch_size=50

def create_model():  
    model=resnet18.make_resnet_18(dim).call(isTrain=False)
    model.trainable=False
    
    optimizer=tf.keras.optimizers.Adam(1e-4)
    
    return model,optimizer

def load_model(checkpoint_path,model):
    #Optimisers
    
    
    #Models
    # model,optimizer= create_model()
    checkpoint_dir = os.path.join(checkpoints, checkpoint_path)
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

# images_concat_shuffled,images_labels_concat_shuffled=data.prep_dataset(dim,test_dir1,test_dir2,13000)
unbias_list = tf.data.Dataset.list_files(unbias_dir + '/*')        
preprocess_function = partial(datat.preprocess_image, target_size=dim)  #Partially fill in a function data.preprocess_image with the arguement image_size
unbias_data = unbias_list.map(preprocess_function).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds

EPOCH=15


if __name__=="__main__":

    # Load gender classifier
    model_classifier,optimizer= create_model() 
    model_classifier=load_model(checkpoint_path,model_classifier)
    
    
    count=0
    #(BaseLine Measruement) Run the D_bias Fairness metric
    for step, (image) in enumerate(unbias_data):
        if count==0:
            # Predict
            predicted_labels=model_classifier(image)
            out=tf.reduce_mean(predicted_labels,axis=0)
            count+=1
        else:
            predicted_labels=model_classifier(image)
            predicted_labels=tf.reduce_mean(predicted_labels,axis=0)     
            out = tf.reduce_mean([out,predicted_labels],axis=0)

    count=0
    epoch=2000
    noise=tf.random.normal([epoch,batch_size,100])
    
    #Load generator
    gen_unweighted_bias=l.load_model("./training_checkpoints_unweighted_biasPoint9_2212")
    
    #Run Generator proGAN metric (Unweighted_bias)
    for i in range(epoch):
        image_ub=gen_unweighted_bias(noise[i])
        if count==0:
            # Predict
            predicted_labels_generator_ub=model_classifier(image_ub)
            out_generator_unweight_ub=tf.reduce_mean(predicted_labels_generator_ub,axis=0)
            count+=1
        else:
            predicted_labels_generator_ub=model_classifier(image_ub)
            predicted_labels_generator_ub=tf.reduce_mean(predicted_labels_generator_ub,axis=0)     
            out_generator_unweight_ub = tf.reduce_mean([out_generator_unweight_ub,predicted_labels_generator_ub],axis=0)
            
            
    loss_metric_unweight=tf.reduce_sum(tf.abs(out-out_generator_unweight_ub))
    
    #Load generator
    gen_weighted_bias=l.load_model("./training_checkpoints_weighted_biasPoint9_2212")
    
    count=0
    #Run Generator Fainess metric (weighted_bias)
    for i in range(epoch):
        image_wb=gen_weighted_bias(noise[i])
        if count==0:
            # Predict
            predicted_labels_generator_wb=model_classifier(image_wb)
            out_generator_weighted_wb=tf.reduce_mean(predicted_labels_generator_wb,axis=0)
            count+=1
        else:
            predicted_labels_generator_wb=model_classifier(image_wb)
            predicted_labels_generator_wb=tf.reduce_mean(predicted_labels_generator_wb,axis=0)     
            out_generator_weighted_wb = tf.reduce_mean([out_generator_weighted_wb,predicted_labels_generator_wb],axis=0)
    
    
    loss_metric_weight=tf.reduce_sum(tf.abs(out-out_generator_weighted_wb))

    #Load generator
    gen_fair=l.load_model("./training_checkpoints_run3_best_unbias")
    
    count=0
    #Run Generator Fainess metric (unbias)
    for i in range(epoch):
        image_xb=gen_fair(noise[i])
        if count==0:
            # Predict
            predicted_labels_generator_xb=model_classifier(image_xb)
            out_generator_fair_xb=tf.reduce_mean(predicted_labels_generator_xb,axis=0)
            count+=1
        else:
            predicted_labels_generator_xb=model_classifier(image_xb)
            predicted_labels_generator_xb=tf.reduce_mean(predicted_labels_generator_xb,axis=0)     
            out_generator_fair_xb = tf.reduce_mean([out_generator_fair_xb,predicted_labels_generator_xb],axis=0)
    
    
    loss_metric_fair=tf.reduce_sum(tf.abs(out-out_generator_fair_xb))

    # # with train_summary_writer_Test.as_default():
    # #     tf.summary.histogram("D_bias_fairness",loss_metric,step=0)
            

    print ("Fairness scores=================================================")
    print ("Fainess Meric (Unweighted): "+str(loss_metric_unweight.numpy()))
    print ("Fainess Meric (Weighted): "+str(loss_metric_weight.numpy()))
    print ("Fainess Meric (Fair): "+str(loss_metric_fair.numpy()))
    print ("Classification Ratios===========================================")
    print ("Average Classification (Data):"+str(out))
    print ("Average Classification (Unweighted):"+str(out_generator_unweight_ub))
    print ("Average Classification (weighted):"+str(out_generator_weighted_wb))
    print ("Average Classification (Fair):"+str(out_generator_fair_xb))
   
   
        
