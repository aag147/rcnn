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
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    #genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 

#if False:
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
#                      metrics={'rpn_out_class':'categorical_accuracy'}) 
    
    # train
    callbacks = [callbacks.MyModelCheckpointInterval(cfg), \
                 callbacks.SaveLog2File(cfg), \
                 callbacks.PrintCallBack()]
    
    model_rpn.fit_generator(generator = genTrain.begin(), \
                steps_per_epoch = genTrain.nb_batches, \
                epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)

    path = cfg.my_weights_path + 'weights-theend.h5'
    if not os.path.exists(path):
        model_rpn.save_weights(path)

    
    print('Path:', cfg.my_results_path)