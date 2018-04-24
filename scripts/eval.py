# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')

import utils
from model_trainer import model_trainer
from generators import DataGenerator
from methods import HO_RCNN
from load_data import data

import numpy as np
import sys



print('Loading data...')
# Load data
data = data()
cfg = data.cfg

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
print('Evaluating model...')
res = trainer.evaluateModel(eval(cfg.testdata))
print("F1:", res.F1, "nb_zeros", res.nb_zeros)
print('Saving Y_Hat...')
utils.save_obj_nooverwrite(res.Y_hat, cfg.my_results_path + 'y_hat')