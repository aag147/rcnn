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
    
    if not os.path.exists(cfg.my_save_path + 'detections/'):
        os.makedirs(cfg.my_save_path + 'detections/')
    cfg.my_save_path += 'detections/'
                
            

if True:
    # models
    loss_cls = losses.rpn_loss_cls(cfg.nb_anchors)
    loss_rgr = losses.rpn_loss_regr(cfg.nb_anchors)
    
    get_custom_objects().update({"rpn_loss_cls_fixed_num": loss_cls})
    get_custom_objects().update({"rpn_loss_regr_fixed_num": loss_rgr})
    
    print('weights path', cfg.my_shared_weights)
    print('results path', cfg.my_save_path)
#    model_rpn = load_model(cfg.my_shared_weights)
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg, mode='train')
    model_rpn.load_weights(cfg.my_shared_weights)

    genIterator = genTrain.begin()
    detMeta = {}
    alltimes = np.zeros((genTrain.nb_batches, 4))
    for batchidx in range(genTrain.nb_batches):        
        X, y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['id']
        
        p_start = time.time()
        x_class, x_deltas = model_rpn.predict_on_batch(X)
        p_end   = time.time()
       
        pp_start = time.time()
        pred_anchors = helper.deltas2Anchors(x_class, x_deltas, cfg, imageDims, do_regr=True)
        pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.rpn_nms_overlap_thresh)
        pred_anchors = pred_anchors[:,0:4]
        bboxes, target_labels, target_deltas, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, data.class_mapping, cfg)
        pp_end   = time.time()

        
        if bboxes is None:
            detMeta[imageID] = None
        else:
            bboxes_clean, target_labels_clean, target_deltas_clean = helper.bboxes2DETformat(bboxes, target_labels, target_deltas, cfg)
            detMeta[imageID] = {'imageName': imageMeta['imageName'], 'rois':bboxes_clean, 'target_props':target_labels_clean, 'target_deltas':target_deltas_clean}
                
        times = list(times) + [p_end-p_start,pp_end-pp_start]
        alltimes[batchidx,:] = times
        utils.update_progress_new(batchidx+1, genTrain.nb_batches, imageMeta['imageName'])
        path = cfg.my_save_path + imageID
        utils.save_dict(detMeta[imageID], path)
    
    print()
    print('Path:', cfg.my_save_path)
    print('Times', np.mean(alltimes, axis=0))