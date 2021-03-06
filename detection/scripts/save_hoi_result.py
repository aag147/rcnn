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
from hoi_generators import DataGenerator

import methods,\
       stages
import hoi_test


if True:
    # Load data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True, mode='test', approach='evalnew')
#    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test', approach='new')
    
    Models = methods.AllModels(cfg, mode='test', do_rpn=False, do_det=False, do_hoi=True)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')
    
sys.stdout.flush()
# Test data
#evalTest, imageMeta = hoi_test.saveEvalData(genTest, Stages, cfg, hoi_mapping)
#hoi_test.saveEvalResults(genTest, cfg, obj_mapping, hoi_mapping)
#import draw
#draw.drawOverlapHOIRes(evalTestSub, genTest.imagesMeta, obj_mapping, hoi_mapping, genTest.images_path)

# Train data
evalTrain, imageMeta = hoi_test.saveEvalData(genTrain, Stages, cfg, hoi_mapping)
hoi_test.saveEvalResults(genTrain, cfg, obj_mapping, hoi_mapping)
