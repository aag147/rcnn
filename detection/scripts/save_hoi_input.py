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
import numpy as np

if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels    
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
#    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)
    

    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)
if True:
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')


inputMeta = hoi_test.saveInputData(genTrain, Stages, cfg)

print()
print('Path:', cfg.my_output_path)
