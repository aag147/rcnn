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

import utils,\
       extract_data,\
       methods,\
       losses,\
       callbacks,\
       filters_helper as helper
from rpn_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
import os
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects


if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    cfg.faster_rcnn_config()

    # data
#    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 
    
    
    def evaluteRPN(x_class, x_deltas, imageMeta, imageDims, cfg):
        pred_anchors = helper.deltas2Anchors(x_class, x_deltas, cfg, imageDims)
#        pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
        
        bboxes = imageMeta['objects']
        scale = imageDims['scale']
        gta = helper.normalizeGTboxes(bboxes, scale=scale, roundoff=True)
        
        true = 0
        false = 0
        total = 0
        
        for anchor in pred_anchors:
            if anchor[4] > 0.5:
                at = {'xmin': anchor[0], 'ymin': anchor[1], 'xmax': anchor[0]+anchor[2], 'ymax': anchor[1]+anchor[3]}
                for gt in gta:
                    iou = utils.get_iou(gt, at)
                    if iou > 0.5:
                        true += 1
                    else:
                        false += 1
                    total += 1
        return true / total, total, true
                
            

if True:
    # models
#    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
#    cfg.my_weights = ''
#    path = cfg.my_weights_path + cfg.my_weights
##    model_rpn.load_weights(path)    
#    model_rpn.compile(optimizer='sgd', loss='mse')
    
    
    loss_cls = losses.rpn_loss_cls(cfg.nb_anchors)
    loss_rgr = losses.rpn_loss_regr(cfg.nb_anchors)
    
    get_custom_objects().update({"rpn_loss_cls_fixed_num": loss_cls})
    get_custom_objects().update({"rpn_loss_regr_fixed_num": loss_rgr})
    
    
    path = cfg.my_weights_path + cfg.my_weights
    model_rpn = load_model(path)
    
    model_rpn.save_weights(cfg.my_weights_path + 'weights.h5')

    genIterator = genVal.begin()
    accs = np.zeros([genVal.nb_batches, 3])
    for batchidx in range(genVal.nb_batches):
        utils.update_progress(batchidx / genVal.nb_batches)
        X, y, imageMeta, imageDims, times = next(genIterator)
        x_class, x_deltas = model_rpn.predict_on_batch(X)
        accs[batchidx,:] = evaluteRPN(x_class, x_deltas, imageMeta, imageDims, cfg)
        
    print(np.mean(accs, axis=0))
    
    print('Path:', cfg.my_results_path)