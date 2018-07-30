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

import utils,\
       extract_data,\
       methods,\
       callbacks,\
       filters_helper as helper
from hoi_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
import tensorflow as tf

if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    utils.saveConfig(cfg)

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)

    # models
    Models = methods.AllModels(cfg, mode='train', do_hoi=True)
    _, _, model_hoi = Models.get_models()
    
    sys.stdout.flush()
    
#if False:    
    # train
    callbacks = [callbacks.MyModelCheckpointInterval(cfg), \
                 callbacks.MyLearningRateScheduler(cfg), \
                 callbacks.MyModelCheckpointWeightsInterval(cfg),\
                 callbacks.SaveLog2File(cfg), \
                 callbacks.PrintCallBack()]

    if cfg.dataset == 'TUPPMI':
        model_hoi.fit_generator(generator = genTrain.begin(), \
                    steps_per_epoch = genTrain.nb_batches, \
                    verbose = 2,\
                    epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)
    else:
        genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=False, mode='val')
        model_hoi.fit_generator(generator = genTrain.begin(), \
                    steps_per_epoch = genTrain.nb_batches, \
                    verbose = 2,\
                    max_queue_size = 100,\
                    validation_data = genTest.begin(), \
                    validation_steps = genTest.nb_batches, \
                    epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)

    # Save stuff
    Models.save_model()

    print('Path:', cfg.my_results_path)