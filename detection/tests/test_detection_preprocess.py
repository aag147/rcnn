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


if True:
    ### Config ###
    data = extract_data.object_data(False)
    cfg = data.cfg
    class_mapping = data.class_mapping    

### test ###
# Get data, image and anchor ground truths
genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val').begin()
X, [Y1,Y2], imageMeta, imageDims = next(genVal)

# filter
X = X[0]
rpn_props = Y1[:,:,:,9:]
rpn_deltas = Y2[:,:,:,36:]

# post preprocessing
pred_anchors = helper.deltas2Anchors(rpn_props, rpn_deltas, cfg, imageDims)
pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
pred_anchors = pred_anchors[:, 0: -1]

# get inputs and targets
rois, true_labels, true_boxes, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, class_mapping, cfg)
norm_rois = filters_detection.prepareInputs(rois, imageDims)

# reduce and filter
samples = helper.reduce_rois(true_labels, cfg)
rois = rois[samples, :]
norm_rois = norm_rois[:, samples, :]
det_props = true_labels[:, samples, :]
det_deltas = true_boxes[:, samples, 320:]

# post preprocessing
pred_boxes = helper.deltas2Boxes(det_props, det_deltas, rois, cfg)
#pred_boxes = helper.non_max_suppression_boxes(pred_boxes, cfg)
import draw
anchors = draw.drawPositiveRois(X, pred_boxes)
bboxes = draw.drawGTBoxes(X, imageMeta, imageDims)
