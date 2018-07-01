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
import filters_detection,\
       filters_helper as helper

import methods,\
       stages,\
       filters_rpn
import utils
import draw
import numpy as np

if False:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True)
#    genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
    
#if True:
#    cfg.use_mean=True
#    cfg.my_results_dir = '80b'
#    cfg.update_paths()
if True:
    cfg.do_fast_hoi = True
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=True)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')

genIterator = genVal.begin()

for i in range(1):
#    X, y, imageMeta, imageDims, times = next(genIterator)
    imageID = imageMeta['imageName'].split('.')[0]

    print('Stage one...')
#    proposals = Stages.stageone([X], y, imageMeta, imageDims)
    print('Stage two...')
#    bboxes = Stages.stagetwo([X,proposals], imageMeta, imageDims)
    print('Stage three...')
    all_hoi_hbboxes, all_hoi_obboxes, all_hoi_props = Stages.stagethree([X,bboxes], imageMeta, imageDims, obj_mapping, include='all')
#    draw.drawOverlapRois((X[0]-np.min(X))/np.max(X), bboxes, imageMeta, imageDims, cfg, obj_mapping)