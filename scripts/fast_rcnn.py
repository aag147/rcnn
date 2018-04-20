# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('..')
sys.path.append('rcnn/')

import utils
from model_trainer import model_trainer
from config import config
from config_helper import set_config
from method_configs import rcnn_hoi_labels as mcfg
from load_data import data
from generators import DataGenerator
from methods import HO_RCNN

import cv2 as cv, numpy as np
import copy as cp


np.seterr(all='raise')

#plt.close("all")

# Read data
if True:
    print('Loading data...')
    # Load data
    data = data()
    cfg = data.cfg
    
if True:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, cfg=cfg, data_type='test')  

if True:    
    # Save config
    utils.saveConfig(cfg)
    utils.saveSplit(cfg, list(data.trainMeta.keys()), list(data.valMeta.keys()))
    
    # Create model
    print('Creating model...')
    model = HO_RCNN(cfg)
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
