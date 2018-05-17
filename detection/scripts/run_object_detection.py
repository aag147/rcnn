# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../fast_rcnn/')
sys.path.append('../cfgs/')
sys.path.append('../layers/')

import numpy as np
import keras

import utils,\
       extract_data,\
       methods,\
       losses,\
       filters_detection,\
       filters_hoi,\
       filters_helper as helper
from fast_generators import DataGenerator
    
# meta data
data = extract_data.data()

# config
cfg = data.cfg
cfg.fast_rcnn_config()

cfg.nb_anchors = 9
cfg.nb_human_classes = 2
cfg.nb_object_classes = 81
cfg.nb_hoi_classes = 600
cfg.nb_rois = 32
cfg.pool_size = 7
cfg.overlap_thresh=0.7,
cfg.max_boxes=300

# models
model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg)

# data
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  


for epochidx in range(cfg.nb_epochs):
    for batchidx in range(cfg.nb_batches):
         
        X, [Y1,Y2], imageMeta, imageDims = next(genTrain)

        loss_rpn = model_rpn.train_on_batch(X, [Y1,Y2])

        [x_class, x_deltas] = model_rpn.predict_on_batch(X)

        rois = rois.rpn_to_roi(x_class, x_deltas, cfg)
        rois, true_labels, true_boxes, IouS = rois.calc_iou(rois, img_data, cfg, data.object_mapping)
        
        samples = rois.reduce_rois(true_labels)
        
        loss_class = model_detection.train_on_batch([X, rois[:, samples, :]],
                                                    [true_labels[:, samples, :], true_boxes[:, samples, :]])