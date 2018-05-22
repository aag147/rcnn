# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')
sys.path.append('../data/')
sys.path.append('../../detection/filters/')


import utils
from model_trainer import model_trainer
from load_data import data
from fast_generators import DataGenerator
from methods import Fast_HO_RCNN

import numpy as np


np.seterr(all='raise')

#plt.close("all")

if True:
    # Load data
    print('Loading data...')
    data = data()
    cfg = data.cfg
    cfg.fast_rcnn_config()
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, labels=data.labels, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, labels=data.labels, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, labels=data.labels, cfg=cfg, data_type='test')  

if True:    
    # Save config
    utils.saveConfig(cfg)
    utils.saveSplit(cfg, list(data.trainMeta.keys()), list(data.valMeta.keys()))
    
    # Create model
    print('Creating model...')
    model = Fast_HO_RCNN(cfg)
    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(cfg)
    
#    from keras.utils import plot_model
#    plot_model(model, to_file='model.png')
    
if True:
    # Train model
    print('Training model...')
    trainer.trainModel(cfg)
    
    # Save stuff
    print('Path:', cfg.my_results_path)
    print('Saving final model...')
    trainer.saveModel(cfg)

    print('Testing model on test...')
    resTest = trainer.evaluateModel(genTest)    
    print("F1 (test!):", resTest.F1, "nb_zeros", resTest.nb_zeros)
    print('Testing model on training...')
    resTrain = trainer.evaluateModel(genTrain)
    print("F1 (train):", resTrain.F1, "nb_zeros", resTrain.nb_zeros)
    
    utils.save_obj_nooverwrite(resTest.Y_hat, cfg.my_results_path + 'y_hat')
    
    f= open(cfg.my_results_path + "tests.txt","a")
    f.close()
    newline = 'test: %.4f, nb_zeros: %.03d | train: %.4f, nb_zeros: %.03d\n' % (resTest.F1, resTest.nb_zeros, resTrain.F1, resTrain.nb_zeros)
    with open(cfg.my_results_path + "tests.txt", 'a') as file:
        file.write(newline)
        


if False:
    testIterator = genTest.begin()
    X,y = next(testIterator)
    
    loss = model.train_on_batch(X, y)
    pred = model.predict_on_batch(X)