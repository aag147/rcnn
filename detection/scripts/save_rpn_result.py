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
import rpn_test



if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
        
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True, mode='test')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True, mode='test')

# Val data    
#evalVal = rpn_test.saveEvalData(genVal, None, cfg)
#GTMeta = rpn_test.saveEvalResults(evalVal, genVal, cfg, obj_mapping)

# Train data
evalTrain = rpn_test.saveEvalData(genTrain, None, cfg)
rpn_test.saveEvalResults(evalTrain, genTrain, cfg, obj_mapping)
