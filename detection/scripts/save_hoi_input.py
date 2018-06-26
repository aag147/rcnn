# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import extract_data
from rpn_generators import DataGenerator

import methods,\
       stages
import hoi_test
import utils

import os

if True:
    # Load data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    if not os.path.exists(cfg.my_save_path + 'detections/'):
        os.makedirs(cfg.my_save_path + 'detections/')
    cfg.my_save_path += 'detections/'    
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    

if True:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')

    inputMeta = hoi_test.saveInputData(genTrain, Stages, cfg)

print()
print('Path:', cfg.my_output_path)
