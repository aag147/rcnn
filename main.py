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
from generators import DataGenerator
from methods import HO_RCNN, HO_RCNN_2

import cv2 as cv, numpy as np
import copy as cp


np.seterr(all='raise')

#plt.close("all")

# Read data
if True:
    # Load data
    cfg = config()
    cfg = set_config(cfg)
    trainMeta = utils.load_dict(cfg.data_path + 'train')
    testMeta = utils.load_dict(cfg.data_path + 'test') 
    labels = utils.load_dict(cfg.data_path + 'labels')
    
    if cfg.max_classes is not None:
        # Reduce data to include only max_classes number of different classes
        trainStats, counts = utils.getLabelStats(trainMeta, labels)
        trainMeta, reduced_idxs = utils.reduceTrainData(trainMeta, counts, cfg.max_classes)
        testMeta = utils.reduceTestData(testMeta, reduced_idxs)
        labels = utils.idxs2labels(reduced_idxs, labels)
        
        
    cfg.nb_classes = len(labels)        
    trainMeta, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
    
if True:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=valMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=testMeta, cfg=cfg, data_type='test')  

if True:    
    # Create model
    model = HO_RCNN(cfg)
    # train model
    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(cfg)
    trainer.trainModel(cfg)
    trainer.saveHistory(cfg)
    res = trainer.evaluateModel(genTest)
    print("F1:", res.F1, "nb_zeros", res.nb_zeros)
