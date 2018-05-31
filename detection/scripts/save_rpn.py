# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import numpy as np
import keras
import time

import utils,\
       extract_data,\
       methods,\
       losses,\
       callbacks,\
       filters_helper as helper,\
       filters_detection
from rpn_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
from keras.models import load_model


from keras.utils.generic_utils import get_custom_objects


import os

if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    class_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
#    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 
    
    
    if not os.path.exists(cfg.my_results_path + 'detections/'):
        os.makedirs(cfg.my_results_path + 'detections/')
    
    
    def getDETData(x_class, x_deltas, imageMeta, imageDims, cfg):
        pred_anchors = helper.deltas2Anchors(x_class, x_deltas, cfg, imageDims, do_regr=True)
        pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
        pred_anchors = pred_anchors[:,0:4]
        
        rois, true_labels, true_boxes, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, data.class_mapping, cfg)
    
        if rois is None:
            return None, None, None
    
        norm_rois = filters_detection.prepareInputs(rois, imageDims)
        
        # reduce and filter
#        samples = helper.reduce_rois(true_labels, cfg)
#        rois = rois[samples, :]
#        norm_rois = norm_rois[:, samples, :]
#        det_props = true_labels[:, samples, :]
#        det_deltas = true_boxes[:, samples, :]
        return norm_rois, true_labels, true_boxes
                
            

if True:
    # models
    loss_cls = losses.rpn_loss_cls(cfg.nb_anchors)
    loss_rgr = losses.rpn_loss_regr(cfg.nb_anchors)
    
    get_custom_objects().update({"rpn_loss_cls_fixed_num": loss_cls})
    get_custom_objects().update({"rpn_loss_regr_fixed_num": loss_rgr})
    
    model_rpn = load_model(cfg.my_shared_weights)

    genIterator = genTrain.begin()
    detMeta = {}
    alltimes = np.zeros((genTrain.nb_batches, 4))
    for batchidx in range(genTrain.nb_batches):        
        X, y, imageMeta, imageDims, times = next(genIterator)
        p_start = time.time()
        x_class, x_deltas = model_rpn.predict_on_batch(X)
        p_end   = time.time()
        pp_start = time.time()
        norm_rois, det_props, det_deltas = getDETData(x_class, x_deltas, imageMeta, imageDims, cfg)
        pp_end   = time.time()
        
        times = list(times) + [p_end-p_start,pp_end-pp_start]
        alltimes[batchidx,:] = times
        
        imageID = imageMeta['id']
        if norm_rois is None:
            detMeta[imageID] = None
        else:
            detMeta[imageID] = {'imageName': imageMeta['imageName'], 'rois':norm_rois.tolist(), 'target_props':det_props.tolist(), 'target_deltas':det_deltas.tolist()}
            
        utils.save_dict(detMeta[imageID], cfg.my_results_path + 'detections/'+imageMeta['imageName'].split('.')[0])
        utils.update_progress_new(batchidx, genTrain.nb_batches, times, imageMeta['imageName'])
    
    print('Path:', cfg.my_results_path)
    print('Times', np.mean(alltimes, axis=0))