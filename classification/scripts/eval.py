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
from generators import DataGenerator
from methods import HO_RCNN
import load_data

import numpy as np
import sys



print('Loading data...')
# Load data
data = load_data.data()
cfg = data.cfg
cfg.rcnn_config()

# Create batch generators
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')   

# Load model
print('Loading model...')
model = HO_RCNN(cfg)
# Evaluate model
trainer = model_trainer(model=model, task=cfg.task)
trainer.compileModel(cfg)

print('Evaluating model on test data...')
resTest = trainer.evaluateModel(genTest)
print("F1 (test!):", resTest.F1, "nb_zeros", resTest.nb_zeros)
print('Saving test Y_Hat...')
utils.save_obj_nooverwrite(resTest.Y_hat, cfg.my_results_path + 'y_hat_test')

print('Evaluating model on training data...')
resTrain = trainer.evaluateModel(genTrain)
print("F1 (train):", resTrain.F1, "nb_zeros", resTrain.nb_zeros)
print('Saving training Y_Hat...')
utils.save_obj_nooverwrite(resTrain.Y_hat, cfg.my_results_path + 'y_hat_train')