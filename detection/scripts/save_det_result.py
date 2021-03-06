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
from det_generators import DataGenerator

import methods,\
       stages
import det_test



if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True, mode='test')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True, mode='test')
#    genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
    Models = methods.AllModels(cfg, mode='test', do_rpn=False, do_det=True, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')

# Val data
evalVal = det_test.saveEvalData(genVal, Stages, cfg, obj_mapping)
det_test.saveEvalResults(genVal, cfg)

# Test data
#evalTest = det_test.saveEvalData(genTest, Stages, cfg, obj_mapping)
#det_test.saveEvalResults(evalTest, genTest, cfg)

# Train data
#evalTrain = det_test.saveEvalData(genTrain, Stages, cfg, obj_mapping)
#det_test.saveEvalResults(evalTrain, genTrain, cfg)
