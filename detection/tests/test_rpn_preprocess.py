# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:26:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import utils,\
       extract_data,\
       methods,\
       losses,\
       filters_detection,\
       filters_hoi,\
       filters_helper as helper
from detection_generators import DataGenerator
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop


if False:
    ### Config ###
    data = extract_data.object_data(False)
    cfg = data.cfg
    

### test ###
# Get data, image and anchor ground truths
genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train').begin()
X, [Y1,Y2], imageMeta, imageDims, times = next(genTrain)
X = X[0]
X -= np.min(X)
X /= 255
X = X[:,:,(2,1,0)]
Y1 = Y1[:,:,:,12:]
Y2 = Y2[:,:,:,48:]
pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims)
pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)

import draw
anchors = draw.drawPositiveAnchors(X, pred_anchors, cfg)
bboxes = draw.drawGTBoxes(X, imageMeta, imageDims)