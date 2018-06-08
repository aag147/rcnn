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
    class_mapping = data.class_mapping
    utils.saveConfig(cfg)

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)

    # models        -0.0135234 -0.013523407
    Models = methods.AllModels(cfg, mode='train', do_rpn=True)
    Models.compile_models()
    model_rpn, _, _ = Models.get_models()

    print(model_rpn.layers[18].get_weights()[0][0,0,0,0])

if True:    
    # train
    callbacks = [callbacks.MyModelCheckpointInterval(cfg), \
                 callbacks.MyLearningRateScheduler(cfg), \
                 callbacks.SaveLog2File(cfg), \
                 callbacks.PrintCallBack()]
    
    model_rpn.fit_generator(generator = genTrain.begin(), \
                steps_per_epoch = genTrain.nb_batches, \
                epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)

    print(model_rpn.layers[18].get_weights()[0][0,0,0,0])

    # Save stuff
    Models.save_model()

    
    print('Path:', cfg.my_results_path)