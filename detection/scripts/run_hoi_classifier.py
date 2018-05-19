# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../filters/')
sys.path.append('../../classification/data/')
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
       filters_helper as helper,\
       load_data
from fast_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam


if True:
    # config
    data = load_data.data()
    cfg = data.cfg
    cfg.fast_rcnn_config()
        
    # labels
    labels = data.labels

    # data
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, labels=data.labels, cfg=cfg, data_type='train')


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


