# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
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
from detection_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
import os

if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    cfg.fast_rcnn_config()

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
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
    cfg.my_weights = ''
    path = cfg.my_weights_path + cfg.my_weights
#    model_rpn.load_weights(path)    
    model_rpn.compile(optimizer='sgd', loss='mse')

    genIterator = genVal.begin()
    accs = np.zeros([genVal.nb_batches, 3])
    for batchidx in range(genVal.nb_batches):
        utils.update_progress(batchidx / genVal.nb_batches)
        X, y, imageMeta, imageDims, times = next(genIterator)
        x_class, x_deltas = model_rpn.predict_on_batch(X)
        accs[batchidx,:] = evaluteRPN(x_class, x_deltas, imageMeta, imageDims, cfg)
        
    print(np.mean(accs, axis=0))
    
    print('Path:', cfg.my_results_path)