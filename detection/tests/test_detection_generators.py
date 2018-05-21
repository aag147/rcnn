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
import time

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

total_times = np.array([0.0,0.0])
for i in range(genTrain.nb_batches):
    try:
        X, y, imageMeta, imageDims, times = next(trainIterator)
        if X is None:
            continue
    except:
        print('error', imageMeta['imageName'])
        break
    total_times += times
    print('finally', i, imageMeta['imageName'])
    break
#    utils.update_progress_new(i, genTrain.nb_batches, list(times) + [0,0], imageMeta['imageName'])
#    print('t',X[0].shape, X[1].shape, y[0].shape, y[1].shape)
#    
    #utils.save_obj(y, cfg.weights_path +'anchors/' + imageMeta['imageName'].split('.')[0])
    s = time.time()
    utils.load_obj(cfg.weights_path +'anchors/' + imageMeta['imageName'].split('.')[0])
    f = time.time()
    print(f-s, times[1])
#    break
print(f-s)

