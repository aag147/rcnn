# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../data/')
sys.path.append('../models/')
sys.path.append('../../shared/')

import utils
from model_trainer import model_trainer
from load_data import data
from generators import DataGenerator
from methods import HO_RCNN, HO_RCNN_tf

import numpy as np


np.seterr(all='raise')

#plt.close("all")

if True:
    # Load data
    print('Loading data...')
    data = data(method='normal')
    cfg = data.cfg
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  

if True:    
    # Save config
    utils.saveConfig(cfg)
    utils.saveSplit(cfg, list(data.trainMeta.keys()), list(data.valMeta.keys()))
    
    # Create model
    print('Creating model...')
    model = HO_RCNN_tf(cfg)
    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(cfg)

if True:   
    # Train model
    print('Training model...')
    trainer.trainModel(cfg)
    
    # Save stuff
    print('Saving final model...')
    trainer.saveModel(cfg)
    print('Testing model on test...')
    resTest = trainer.evaluateModel(genTest)    
    print("F1 (test!):", resTest.F1, "nb_zeros", resTest.nb_zeros, "mAP", resTest.mAP)
    print('Testing model on training...')
    resTrain = trainer.evaluateModel(genTrain)
    print("F1 (train):", resTrain.F1, "nb_zeros", resTrain.nb_zeros, "mAP", resTrain.mAP)

    utils.save_obj_nooverwrite(resTest.Y_hat, cfg.my_results_path + 'y_hat')  
    utils.save_obj_nooverwrite(resTest.Y, cfg.my_results_path + 'y_test')
    
    f= open(cfg.my_results_path + "tests.txt","a")
    f.close()
    newline = 'epoch:%0.3d :: test: %.4f, nb_zeros: %.03d, %.4f | train: %.4f, nb_zeros: %.03d, %.4f\n' % \
        (cfg.epoch_end, resTest.F1, resTest.nb_zeros, resTest.mAP, resTrain.F1, resTrain.nb_zeros, resTrain.mAP)
    with open(cfg.my_results_path + "tests.txt", 'a') as file:
        file.write(newline)
