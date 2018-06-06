# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

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
from rpn_generators import DataGenerator
import filters_helper as helper

import numpy as np
import utils
import time
import cv2 as cv
import copy as cp

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')

trainIterator = genTrain.begin()

alltimes = np.zeros((genTrain.nb_batches, 4))
for batchidx in range(genTrain.nb_batches):
    X, [Y1,Y2,M], imageMeta, imageDims, times = next(trainIterator)
    
    
    times = list(times) + [0,0]
    alltimes[batchidx,:] = times
    
    target_labels, target_deltas, val_map = helper.bboxes2RPNformat(Y1, Y2, M, cfg)
    rpnMeta = {'target_labels': target_labels, 'target_deltas': target_deltas, 'val_map': val_map}
    utils.save_obj(rpnMeta, cfg.data_path +'anchors/train/' + imageMeta['imageName'].split('.')[0])
    utils.update_progress_new(batchidx, genTrain.nb_batches, imageMeta['imageName'])
    break

print('Times', np.mean(alltimes, axis=0))
