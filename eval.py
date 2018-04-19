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
from methods import HO_RCNN

import numpy as np
import copy as cp
import sys, getopt


def get_args(cfg):
   try:
      argv = sys.argv[1:]
      opts, args = getopt.getopt(argv,"m:")
   except getopt.GetoptError:
      print('.py -m <my_model>')
      sys.exit(2)
 
   for opt, arg in opts:
      if opt == '-m':
         cfg.my_weights = arg
   return cfg

cfg = config()
cfg = set_config(cfg)
cfg = get_args(cfg)

# Load data 
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

# Create batch generators
genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, data_type='train')
genVal = DataGenerator(imagesMeta=valMeta, cfg=cfg, data_type='val')
genTest = DataGenerator(imagesMeta=testMeta, cfg=cfg, data_type='test')  

# Load model
print('Loading model...')
model = HO_RCNN(cfg)
# Evaluate model
trainer = model_trainer(model=model, task=cfg.task)
trainer.compileModel(cfg)
print('Evaluating model...')
res = trainer.evaluateModel(genTest)
print("F1:", res.F1, "nb_zeros", res.nb_zeros)

