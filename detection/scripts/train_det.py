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
       losses,\
       callbacks,\
       filters_helper as helper
from det_generators import DataGenerator
    


if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    obj_mapping = data.class_mapping
    utils.saveConfig(cfg)

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=False)

    # models
    Models = methods.AllModels(cfg, mode='train', do_det=True)
    _, model_det, _ = Models.get_models()
    

if False:    
    # train
    callbacks = [callbacks.MyModelCheckpointInterval(cfg), \
                 callbacks.MyLearningRateScheduler(cfg), \
                 callbacks.SaveLog2File(cfg), \
                 callbacks.PrintCallBack()]
    
    model_det.fit_generator(generator = genTrain.begin(), \
                steps_per_epoch = genTrain.nb_batches, \
                validation_data = genVal.begin(), \
                validation_steps = genVal.nb_batches, \
                epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)

    # Save stuff
    Models.save_model(saveShared=True)
    
    print('Path:', cfg.my_results_path)