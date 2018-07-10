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



if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    obj_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=False)

    # models
    Models = methods.AllModels(cfg, mode='train', do_rpn='rpn' in cfg.my_results_dir, do_det='det' in cfg.my_results_dir, do_hoi='hoi' in cfg.my_results_dir)
    sys.stdout.flush()

#if False:    
    # Save stuff
    Models.save_model(only_weights=True)
    
    print('Path:', cfg.my_results_path)