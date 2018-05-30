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
    utils.saveConfig(cfg)

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)
    #genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 

    # models
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
    
    print('Obj. classes', cfg.nb_object_classes)
    if cfg.optimizer == 'adam':
        print('Adam opt', 'lr:', cfg.init_lr)
        opt = Adam(lr = cfg.init_lr)
    else:
        print('SGD opt', 'lr:', cfg.init_lr)
        opt = SGD(lr = cfg.init_lr, momentum = 0.9, decay = 0.0005, nesterov=False)
        
    if cfg.rpn_uniform_sampling:
        print('Uniform anchor sampling')
    else:
        print('Non-Uniform anchor sampling')
        
    if cfg.use_channel_mean:
        print('Channel mean preprocessing')
    else:
        print('Image mean/std preprocessing')
        
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        if cfg.use_shared_cnn:
            print('Loading shared weights...')
            model_rpn.load_weights(cfg.my_shared_weights, by_name=True)
            # Only train unique layers
            for i, layer in enumerate(model_rpn.layers):
                layer.trainable = False
                if i > 17:
                    break
        else:
            print('Loading my weights...')
            loss_cls = losses.rpn_loss_cls(cfg.nb_anchors)
            loss_rgr = losses.rpn_loss_regr(cfg.nb_anchors)
            
            get_custom_objects().update({"rpn_loss_cls_fixed_num": loss_cls})
            get_custom_objects().update({"rpn_loss_regr_fixed_num": loss_rgr})
            
            model_rpn = load_model(cfg.my_shared_weights)     
    
    model_rpn.compile(optimizer=opt,\
                      loss=[losses.rpn_loss_cls(cfg.nb_anchors), losses.rpn_loss_regr(cfg.nb_anchors)])
#                      metrics={'rpn_out_class':'categorical_accuracy'}) 
    

if True:    
    # train
    callbacks = [callbacks.MyModelCheckpointInterval(cfg), \
                 callbacks.SaveLog2File(cfg), \
                 callbacks.PrintCallBack()]
    
    model_rpn.fit_generator(generator = genTrain.begin(), \
                steps_per_epoch = genTrain.nb_batches, \
                epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)

    # Save stuff
    for i in range(10):
        modelpath = cfg.my_weights_path + 'model-theend%d.h5' % i
        weightspath = cfg.my_weights_path + 'weights-theend%d.h5' % i
        if not os.path.exists(modelpath):
            model_rpn.save(modelpath)
            model_rpn.save_weights(weightspath)
            break

    
    print('Path:', cfg.my_results_path)