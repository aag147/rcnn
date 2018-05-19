# -*- coding: utf-8 -*-
"""
Created on Sat May 19 10:43:44 2018

@author: aag14
"""


import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../filters/')
sys.path.append('../data/')
sys.path.append('../models/')

import numpy as np
import keras
import os

import utils,\
       callbacks,\
       extract_data,\
       methods,\
       losses,\
       filters_detection,\
       filters_hoi,\
       filters_helper as helper
from detection_generators import DataGenerator
    


if True:
    # meta data
    data = extract_data.object_data()

    # config
    cfg = data.cfg
    cfg.fast_rcnn_config()
    
    # labels
    class_mapping = data.class_mapping

    # data
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')


if True:
    # models
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg, mode='test')
        
    print('Obj. classes', cfg.nb_object_classes)

    print('Loading my weights...')
    path = cfg.my_weights_path + cfg.my_weights
    assert os.path.exists(path), 'invalid path: %s' % path
    model_rpn.load_weights(path, by_name=True)
    model_detection.load_weights(path, by_name=True)
    
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_detection.compile(optimizer='sgd', loss='mse')

if True:

    dataIterator = genVal.begin()
    for epochidx in range(cfg.epoch_end):
        for batchidx in range(genVal.nb_batches):
            X, [Y1,Y2], imageMeta, imageDims = next(dataIterator)
            
            utils.update_progress(batchidx / genVal.nb_batches)
    
            [x_class, x_deltas, features] = model_rpn.predict_on_batch(X)
    
            # post preprocessing
            pred_anchors = helper.deltas2Anchors(x_class, x_deltas, cfg, imageDims)
            pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
            rois = pred_anchors[:, 0: -1]
    
            norm_rois = filters_detection.prepareInputs(rois, imageDims)

            
            det_props, det_deltas = model_detection.predict_on_batch([features, norm_rois])
            pred_boxes = helper.deltas2Boxes(det_props, det_deltas, rois, cfg)
        utils.update_progress(genVal.nb_batches / genVal.nb_batches)