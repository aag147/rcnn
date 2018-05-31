# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys

sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../layers/')
sys.path.append('../data/')
sys.path.append('../../detection/filters/')


from model_trainer import model_trainer
from load_data import data
from old_generators import DataGenerator
import losses
from methods import HO_RCNN, HO_RCNN_OLD
import utils
#import draw


#from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam


import numpy as np
import cv2 as cv
import copy as cp


np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = data(False, method='normal')
    cfg = data.cfg
    cfg.order_of_dims = [2,0,1]
    cfg.par_order_of_dims = [0,1,2,3]
    
    class_mapping = data.class_mapping
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')
    
    trainIterator = genTest.begin()
    
if True:
    print('Loading models...')
    model_tf_all = HO_RCNN(cfg)
    model_th_all = HO_RCNN_OLD(cfg)
    
    model_tf = Model(inputs=model_tf_all.input, outputs=model_tf_all.layers[26].output)  
    model_th = Model(inputs=model_th_all.input, outputs=model_th_all.layers[26].output)
    

    model_tf.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    model_th.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    
    w_tf = model_tf.layers[1].get_weights()
    w_th = model_th.layers[1].get_weights()
  

if True:
    trainer_tf = model_trainer(model=model_tf_all, genTrain=genTrain, genVal=None, genTest=None, task=cfg.task)
    trainer_tf.compileModel(cfg)
    
    trainer_th = model_trainer(model=model_th_all, genTrain=genTrain, genVal=None, genTest=None, task=cfg.task)
    trainer_th.compileModel(cfg)
    
    
    trainer_th.trainModel(cfg)
    print('Testing model on test...')
    resTest = trainer_th.evaluateModel(genTest)    
    print("F1 (test!):", resTest.F1, "nb_zeros", resTest.nb_zeros)

j = 0
for i in range(1):
    X, y = next(trainIterator)

    img = cp.copy(X[0])
    img += 1.0
    img /= 2.0
    
    pred_th = model_th_all.predict_on_batch(X)
    X = [X[0].transpose([0,2,3,1])]
    pred_tf = model_tf_all.predict_on_batch(X)
#    pred_tf = pred_tf.transpose([0,3,1,2])
    diff = pred_tf - pred_th
    s = np.sum(diff)
    print('sum',s)
#    plt.imshow(pred_tf[0,:,:,:])
#    plt.imshow(pred_th[0,:,:,:])

    
    
    
    