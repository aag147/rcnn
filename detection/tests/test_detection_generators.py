# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')


import extract_data
from detection_generators import DataGenerator

import numpy as np
import utils

np.seterr(all='raise')

#plt.close("all")

# Load data
print('Loading data...')
data = extract_data.object_data(False)
cfg = data.cfg
cfg.fast_rcnn_config()

# Create batch generators
genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')


trainIterator = genTrain.begin()

for i in range(genTrain.nb_batches):
#    X, y = next(trainIterator)
    utils.update_progress(i / genTrain.nb_images)
#    print('t',X[0].shape, X[1].shape, y[0].shape, y[1].shape)
#    
#    if i > 100:
#        break