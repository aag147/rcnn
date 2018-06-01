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
    data = data()
    cfg = data.cfg
    cfg.rcnn_config()
    
    
    cfg.order_of_dims = [0,1,2]
    cfg.par_order_of_dims = [0,2,3,1]
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  

if True:    
    # Save config
#    utils.saveConfig(cfg)
#    utils.saveSplit(cfg, list(data.trainMeta.keys()), list(data.valMeta.keys()))
    
    # Create model
    print('Creating model...')
    model = HO_RCNN_tf(cfg)
    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(cfg)
    
    # Train model
    print('Training model...')
    trainer.trainModel(cfg)
    
    # Save stuff
    print('Path:', cfg.my_results_path)
    print('Saving final model...')
    trainer.saveModel(cfg)
    trainer.saveHistory(cfg)
    print('Testing model on test...')
    resTest = trainer.evaluateModel(genTest)
    print("F1 (test!):", resTest.F1, "nb_zeros", resTest.nb_zeros)
    print('Testing model on training...')
    resTrain = trainer.evaluateModel(genTrain)
    print("F1 (train):", resTrain.F1, "nb_zeros", resTrain.nb_zeros)
