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


plt.close("all")
cfg = config()
cfg = set_config(cfg)

# Load data 
trainMeta = utils.load_dict(cfg.data_path+'_train')
testMeta = utils.load_dict(cfg.data_path+'_test') 
trainMeta, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)

# Create batch generators
genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, data_type='train')
genVal = DataGenerator(imagesMeta=valMeta, cfg=cfg, data_type='val')
genTest = DataGenerator(imagesMeta=testMeta, cfg=cfg, data_type='test')  
