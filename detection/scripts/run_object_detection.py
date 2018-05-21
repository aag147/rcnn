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
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
import time

if True:
    # meta data
    data = extract_data.object_data()

    # config
    cfg = data.cfg
    cfg.fast_rcnn_config()
    utils.saveConfig(cfg)
    f= open(cfg.my_results_path + "history.txt","w+")
    f.close()
    
    # labels
    class_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')


if True:
    # models
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
        
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
    model_all.compile(optimizer='sgd', loss='mae')


if True:
    
    my_losses = np.zeros([cfg.epoch_end, genTrain.nb_batches, 4])
    dataIterator = genTrain.begin()
    for epochidx in range(cfg.epoch_end):
        for batchidx in range(genTrain.nb_batches):
            rpn_s = time.time()
            X, [Y1,Y2], imageMeta, imageDims, times = next(dataIterator)
            rpn_f = time.time()
            
            utils.update_progress_new(batchidx, genTrain.nb_batches, my_losses[epochidx, max(batchidx-1,0),:], imageMeta['imageName'])

            loss_rpn = model_rpn.train_on_batch(X, [Y1,Y2])
    
            [x_class, x_deltas] = model_rpn.predict_on_batch(X)
    
            # post preprocessing
            det1_s = time.time()
            pred_anchors = helper.deltas2Anchors(x_class, x_deltas, cfg, imageDims)
            pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
            pred_anchors = pred_anchors[:, 0: -1]
            
            det1_f = time.time()
            det2_s = time.time()
            
            # get inputs and targets
            rois, true_labels, true_boxes, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, class_mapping, cfg)
    
            if rois is None:
                continue
    
            norm_rois = filters_detection.prepareInputs(rois, imageDims)
            
            det2_f = time.time()
            det3_s = time.time()
    
            # reduce and filter
            samples = helper.reduce_rois(true_labels, cfg)
            rois = rois[samples, :]
            norm_rois = norm_rois[:, samples, :]
            det_props = true_labels[:, samples, :]
            det_deltas = true_boxes[:, samples, :]
            
            det3_f = time.time()
            
            loss_class = model_detection.train_on_batch([X, norm_rois],
                                                        [det_props, det_deltas])
#            my_losses[epochidx, batchidx, :] = [loss_rpn[1], loss_rpn[2], loss_class[1], loss_class[2]]
            my_losses[epochidx, batchidx, :] = [times[1], det1_f-det1_s, det2_f-det2_s, det3_f-det3_s]
            
            newline = '%.01d, %.03d, %.4f, %.4f, %.4f, %.4f\n' % \
                (epochidx, batchidx, loss_rpn[1], loss_rpn[2], loss_class[1], loss_class[2])

            with open(cfg.my_results_path + "history.txt", 'a') as file:
                file.write(newline)
            
        utils.update_progress_new(genTrain.nb_batches, genTrain.nb_batches, my_losses[epochidx,-1,:], 'theend')
        
        path = cfg.my_weights_path + 'weights-%d-end.h5' % batchidx
        if not os.path.exists(path):
            model_all.save_weights(path)
            
    
utils.save_obj_nooverwrite(my_losses, path)      
print('Path:', cfg.my_results_path)