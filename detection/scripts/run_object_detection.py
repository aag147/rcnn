# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

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

import utils,\
       callbacks,\
       extract_data,\
       methods,\
       losses,\
       filters_detection,\
       filters_hoi,\
       filters_helper as helper
from detection_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam


if True:
    # meta data
    data = extract_data.object_data()

    # config
    cfg = data.cfg
    cfg.fast_rcnn_config()
    
    # labels
    class_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')


if True:
    # models
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg)
        
    print('Obj. classes', cfg.nb_object_classes)
    if cfg.optimizer == 'adam':
        print('Adam opt', 'lr:', cfg.init_lr)
        opt = Adam(lr = cfg.init_lr)
    else:
        print('SGD opt', 'lr:', cfg.init_lr)
        opt = SGD(lr = cfg.init_lr, momentum = 0.9, decay = 0.0005, nesterov=False)
    
    model_rpn.compile(optimizer=opt,\
                      loss=[losses.rpn_loss_cls(cfg.nb_anchors), losses.rpn_loss_regr(cfg.nb_anchors)])
    model_detection.compile(optimizer=opt,
                            loss=[losses.class_loss_cls, losses.class_loss_regr(cfg.nb_object_classes - 1)],
                            metrics={'det_out_class': 'categorical_accuracy'})

if True:
    dataIterator = genTrain.begin()
    for epochidx in range(cfg.epoch_end):
        for batchidx in range(genTrain.nb_batches):
            utils.update_progress_new(batchidx, genTrain.nb_batches)
            X, [Y1,Y2], imageMeta, imageDims = next(dataIterator)
    
            loss_rpn = model_rpn.train_on_batch(X, [Y1,Y2])
    
            [x_class, x_deltas] = model_rpn.predict_on_batch(X)
    
            # post preprocessing
            pred_anchors = helper.deltas2Anchors(x_class, x_deltas, cfg, imageDims)
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
            det_deltas = true_boxes[:, samples, :]
            
            loss_class = model_detection.train_on_batch([X, norm_rois],
                                                        [det_props, det_deltas])
        utils.update_progress_new(genTrain.nb_batches, genTrain.nb_batches)