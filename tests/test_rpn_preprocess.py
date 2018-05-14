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
       filters_hoi
from detection_generators import DataGenerator
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop


if False:
    ### Config ###
    data = extract_data.object_data(False)
    cfg = data.cfg
    
    # rpn filters
    cfg.rpn_stride = 16
    
    cfg.anchor_sizes = [128, 256, 512]
    cfg.anchor_ratios = [[1, 1], [1, 2], [2, 1]]
    
    cfg.rpn_min_overlap = 0.1
    cfg.rpn_max_overlap = 0.5
    
    # detection filters
    cfg.detection_max_overlap = 0.5
    cfg.detection_min_overlap = 0.1
    cfg.nb_detection_rois = 16
    cfg.nms_max_boxes=300
    cfg.nms_overlap_tresh=0.9
    
    # hoi filters
    cfg.hoi_bbox_threshold = 0.5
    
    # models
    cfg.nb_anchors = len(cfg.anchor_sizes) * len(cfg.anchor_ratios)
    cfg.pool_size = 7
    cfg.nb_object_classes = 81
    cfg.nb_hoi_classes = 601
    
    ### Data ###
    valGTMeta = utils.load_dict(cfg.data_path + 'val_GT')
    class_mapping = utils.load_dict(cfg.data_path + 'class_mapping')
    
#if True:   
    ### models ###
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg)
    optimizer_rpn = Adam(lr=1e-5)
    optimizer_dect = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer_rpn,
                      loss=[losses.rpn_loss_cls(cfg.nb_anchors), losses.rpn_loss_regr(cfg.nb_anchors)])
    model_detection.compile(optimizer=optimizer_dect,
                             loss=[losses.class_loss_cls, losses.class_loss_regr((cfg.nb_object_classes) - 1)],
                             metrics={'dense_class_{}'.format((cfg.nb_object_classes)): 'accuracy'})
    

### test ###
if False:
    # rpn #
    genVal = DataGenerator(imagesMeta = valGTMeta, cfg=cfg, data_type='val').begin()
    X, [Y1,Y2], imageMeta, imageDims = next(genVal)
    
    loss_rpn = model_rpn.train_on_batch(X, [Y1,Y2])
    x_class,x_deltas = model_rpn.predict_on_batch(X)
    
    # detection #
    rois_org = filters_detection.deltas_to_roi(x_class, x_deltas, cfg)
    rois, true_labels, true_boxes, IouS = filters_detection.detection_ground_truths(rois_org, imageMeta, imageDims, class_mapping, cfg)
    samples = filters_detection.reduce_rois(true_labels, cfg)
    
    roisRedux = rois[:, samples, :]
    true_labelsRedux = true_labels[:, samples, :]
    true_boxesRedux = true_boxes[:, samples, :]
    
    loss_class = model_detection.train_on_batch([X, roisRedux], [true_labelsRedux, true_boxesRedux])
    object_scores, object_deltas = model_detection.predict_on_batch([X, roisRedux])

if True:
    # hoi #
    pred_labels, pred_bboxes = filters_hoi.deltas_to_bb(object_scores, object_deltas, rois, cfg)
    filters_hoi.hoi_ground_truths(pred_labels, pred_bboxes, imageMeta, imageDims, cfg)


