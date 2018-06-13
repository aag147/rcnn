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
import time

import utils,\
       extract_data,\
       methods,\
       stages,\
       losses,\
       callbacks,\
       filters_helper as helper,\
       filters_detection
from rpn_generators import DataGenerator

import os

if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 
    
    if not os.path.exists(cfg.my_save_path + 'detections/'):
        os.makedirs(cfg.my_save_path + 'detections/')
    cfg.my_save_path += 'detections/'
                

if True:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=False, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')

genIterator = genVal.begin()
detMeta = {}
        
for batchidx in range(genVal.nb_batches):

    X, y, imageMeta, imageDims, times = next(genIterator)    
    utils.update_progress_new(batchidx+1, genVal.nb_batches, imageMeta['imageName'])
    
    #STAGE 1
    proposals = Stages.stageone(X, y, imageMeta, imageDims)
    
    #STAGE 2
    proposals, target_labels, target_deltas = Stages.stagetwo(proposals, imageMeta, imageDims, include='pre')

    #CONVERT
    if proposals is None:
        detMeta = None
    else:
        detMeta = filters_detection.convertData([proposals, target_labels, target_deltas], cfg)
            
    utils.save_obj(detMeta, cfg.my_save_path + imageMeta['imageName'].split('.')[0])    
    
print()
print('Path:', cfg.my_save_path)
print('Times', np.mean(alltimes, axis=0))