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
    cfg.inputs = [1,0,0]
    
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
    model = Fast_HO_RCNN(cfg)
    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(cfg)
    
    # Train model
    print('Training model...')
    trainer.trainModel(cfg)
    
    # Save stuff
    print('Saving final model...')
    trainer.saveModel(cfg)
    print('Testing model...')
    res = trainer.evaluateModel(genTest)
    print("F1:", res.F1, "nb_zeros", res.nb_zeros)
    utils.save_obj_nooverwrite(res.Y_hat, cfg.my_results_path + 'y_hat')
    print('Path:', cfg.my_results_path)