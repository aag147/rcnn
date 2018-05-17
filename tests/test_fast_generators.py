# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../classification/data/')
sys.path.append('../classification/models/')

from model_trainer import model_trainer
from load_data import data
from fast_generators import DataGenerator
from methods import Fast_HO_RCNN
import utils

import numpy as np


np.seterr(all='raise')

#plt.close("all")

# Load data
print('Loading data...')
data = data()
cfg = data.cfg
cfg.fast_rcnn_config()

# Create batch generators
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')


trainIterator = genTrain.begin()

for i in range(genTrain.nb_batches):
    next(trainIterator)
    utils.update_progress(i / genTrain.nb_images)