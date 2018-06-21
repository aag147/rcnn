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

import extract_data
from rpn_generators import DataGenerator

import methods,\
       stages
import det_test
import os

if True:
    # meta data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    cfg.my_save_path += cfg.dataset + '/rpn' + cfg.my_results_dir + '/detections/'
    if not os.path.exists(cfg.my_save_path):
        os.makedirs(cfg.my_save_path)
    print(cfg.my_save_path)

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 
                

if True:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=False, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')

    det_test.saveInputData(genTrain, Stages, cfg)
    det_test.saveInputData(genVal, Stages, cfg)
    
print()
print('Path:', cfg.my_save_path)
