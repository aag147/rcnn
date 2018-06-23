# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:26:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import extract_data
import utils
from rpn_generators import DataGenerator
import numpy as np

if True:
    ### Config ###
    data = extract_data.object_data(False)
    cfg = data.cfg
    cfg.nb_rpn_proposals = 500
    

### test ###
# Get data, image and anchor ground truths
genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')

genIter = genTrain.begin()

nb_zeros = 0

for i, (imageID, imageMeta) in enumerate(data.trainGTMeta.items()):
    objs = imageMeta['objects']
    if len(objs) == 0:
        nb_zeros += 1
    if i % 1000 == 0:
        utils.update_progress_new(i+1, genTrain.nb_batches, imageID + ': ' + imageID)


counts = np.zeros((genTrain.nb_batches))
for i in range(genTrain.nb_batches):
    break
    X, [Y1,Y2], imageMeta, imageDims, times = next(genIter)
    imageID = imageMeta['imageName'].split('.')[0]

    deltas = Y2[:,:,:,:48]
    nb_pos = np.sum(deltas) / 4
    counts[i] = nb_pos
    
    if i % 1000 == 0:
        utils.update_progress_new(i+1, genTrain.nb_batches, imageID + ': ' + str(nb_pos))
    
    
nb_max = np.max(counts)
nb_min = np.min(counts)
nb_mean = np.mean(counts)
print('max', nb_max)
print('min', nb_min)
print('mean', nb_mean)