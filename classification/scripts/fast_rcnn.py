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

import utils
from model_trainer import model_trainer
import load_data
from fast_generators import DataGenerator
from methods import Fast_HO_RCNN

import numpy as np


np.seterr(all='raise')

#plt.close("all")

if True:
    # Load data
    print('Loading data...')
    data = load_data.data()
    cfg = data.cfg
    cfg.fast_rcnn_config()
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train', labels=data.labels)
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val', labels=data.labels)
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test', labels=data.labels)
    
if True:
    # Save config
    utils.saveConfig(cfg)
    utils.saveSplit(cfg, list(data.trainMeta.keys()), list(data.valMeta.keys()))
    
    # Create model
    print('Creating model...')
    model = Fast_HO_RCNN(cfg)

    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(cfg)

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
    newline = 'epoch:%0.3d :: test: %.4f, nb_zeros: %.03d | train: %.4f, nb_zeros: %.03d\n' % \
        (cfg.epoch_end, resTest.F1, resTest.nb_zeros, resTrain.F1, resTrain.nb_zeros)
    with open(cfg.my_results_path + "tests.txt", 'a') as file:
        file.write(newline)
