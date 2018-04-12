# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('..')
sys.path.append('rcnn/')

import extractTUHOIData as tuhoi
import utils, draw
from model_trainer import model_trainer
from config import config
from config_helper import set_config
from models import AlexNet, PairWiseStream
from generators import DataGenerator
from methods import HO_RCNN, HO_RCNN_2

from matplotlib import pyplot as plt
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
         cfg.my_model = arg
   return cfg

plt.close("all")
cfg = config()
cfg = set_config(cfg)
cfg = get_args(cfg)

# Load data 
trainMeta = utils.load_dict(cfg.data_path+'_train')
testMeta = utils.load_dict(cfg.data_path+'_test') 
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
print('Evaluate model...')
res = trainer.evaluateModel(genTest)
print("F1:", res.F1, "nb_zeros", res.nb_zeros)

