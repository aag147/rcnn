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

import numpy as np
import utils
import time
import draw
import filters_helper as helper
import methods
import os
import filters_detection
import cv2 as cv
import copy as cp

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    cfg.faster_rcnn_config()
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    
if True:
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        print('Loading my weights...')
        path = cfg.my_weights_path + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        model_rpn.load_weights(path) 

trainIterator = genTrain.begin()

total_times = np.array([0.0,0.0])
j = 0
for i in range(genTrain.nb_batches):
    if True:
        X, y, imageMeta, imageDims, times = next(trainIterator)
#        X, y, imageMeta, imageDims, times = next(trainIterator)
    if True:
        img = cp.copy(X[0])
        img += 1.0
        img /= 2.0

    if False:
        Y1 = y[0][:,:,:,12:];
        idxs = y[0][:,:,:,:12]; idxs = idxs.reshape((-1)); idxs = np.where(idxs)[0]
        
        Y2 = y[1][:,:,:,48:]
                
        props = np.reshape(Y1,(-1,1))
    
    Y1, Y2 = model_rpn.predict_on_batch(X)
    pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims, do_regr=True)
#    pred_anchors = pred_anchors[idxs,:]
    print('positives', np.sum(Y1>0.5))
    print(pred_anchors.shape)
#    draw.drawPositiveAnchors(img, pred_anchors, cfg)
    pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
    print('positives', np.sum(pred_anchors[:,4]>0.5))
    print(pred_anchors.shape)
    draw.drawAnchors(img, pred_anchors, cfg)
    draw.drawGTBoxes(img, imageMeta, imageDims)
    
    rois, true_labels, true_boxes, IouS = filters_detection.prepareTargets(pred_anchors[:,0:4], imageMeta, imageDims, data.class_mapping, cfg)
    print('positives', np.sum(true_labels[:,:,1:]))
    if rois is None:
        continue

    norm_rois = filters_detection.prepareInputs(rois, imageDims)
    boxes = helper.deltas2Boxes(true_labels, true_boxes[:,:,(cfg.nb_object_classes-1)*4:]*0.0, rois, cfg)
    draw.drawAnchors(img, boxes, cfg)
    
    # reduce and filter
    for k in range(2):
        samples = helper.reduce_rois(true_labels, cfg)
#        samples = list(range(cfg.detection_nms_max_boxes))
        rois2 = rois[samples, :]
        det_props2 = true_labels[:, samples, :]
        det_deltas2 = true_boxes[:, samples, (cfg.nb_object_classes-1)*4:]*0.0
    
        boxes = helper.deltas2Boxes(det_props2, det_deltas2, rois2, cfg)    
        draw.drawAnchors(img, boxes, cfg)
#    
#    utils.save_obj(y, cfg.data_path +'anchors/val/' + imageMeta['imageName'].split('.')[0])
#    s = time.time()
#    utils.load_obj(cfg.data_path +'anchors/val/' + imageMeta['imageName'].split('.')[0])
#    f = time.time()
#    print(f-s, times[1])
    if j == 0:
        break
    j += 1
#    break
#print(f-s)

