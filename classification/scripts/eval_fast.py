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
import sys



print('Loading data...')
data = load_data.data()
cfg = data.cfg
cfg.fast_rcnn_config()

# Create batch generators
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train', labels=data.labels)
genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val', labels=data.labels)
genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test', labels=data.labels)  

# Create model
print('Creating model...')
model = Fast_HO_RCNN(cfg)

trainer = model_trainer(model=model, task=cfg.task)
trainer.compileModel(cfg)

print('Evaluating model on test data...')
resTest = trainer.evaluateModel(genTest)
print("F1 (test!):", resTest.F1, "nb_zeros", resTest.nb_zeros)
print('Saving test Y_Hat...')
utils.save_obj_nooverwrite(resTest.Y_hat, cfg.my_results_path + 'y_hat_test')
utils.save_obj_nooverwrite(resTest.Y, cfg.my_results_path + 'y_test')

print('Evaluating model on training data...')
resTrain = trainer.evaluateModel(genTrain)
print("F1 (train):", resTrain.F1, "nb_zeros", resTrain.nb_zeros)
print('Saving training Y_Hat...')
utils.save_obj_nooverwrite(resTrain.Y_hat, cfg.my_results_path + 'y_hat_train')
utils.save_obj_nooverwrite(resTest.Y, cfg.my_results_path + 'y_train')