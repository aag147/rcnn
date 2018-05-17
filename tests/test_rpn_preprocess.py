# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:26:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../detection_rcnn/data/')
sys.path.append('../detection_rcnn/filters/')
sys.path.append('../detection_rcnn/models/')

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


if True:
    ### Config ###
    data = extract_data.object_data(False)
    cfg = data.cfg
    
    # rpn filters
    cfg.rpn_stride = 16
    
    cfg.anchor_sizes = [128, 256, 512]
    cfg.anchor_ratios = [[1, 1], [1, 2], [2, 1]]
    
    cfg.rpn_min_overlap = 0.1
    cfg.rpn_max_overlap = 0.5
    
    cfg.nms_max_boxes=300
    cfg.nms_overlap_thresh=0.9
    

### test ###
# Get data, image and anchor ground truths
genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val').begin()
X, [Y1,Y2], imageMeta, imageDims = next(genVal)
X = X[0]
Y1 = Y1[:,:,:,9:]
Y2 = Y2[:,:,:,36:]
pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims)
pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
import draw
anchors = draw.drawPositiveAnchors(X, pred_anchors)
bboxes = draw.drawGTBoxes(X, imageMeta, imageDims)