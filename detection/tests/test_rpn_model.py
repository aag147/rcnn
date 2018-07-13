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
import cv2 as cv

if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)
#    genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
    
#if True:
#    cfg.use_mean=True
#    cfg.my_results_dir = '80b'
#    cfg.update_paths()
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=False, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')


genIterator = genVal.begin()

for i in range(1):
#    X, y, imageMeta, imageDims, times = next(genIterator)
    imageID = imageMeta['imageName'].split('.')[0]
    
    X, imageDims = filters_rpn.prepareInputs(imageMeta, genVal.images_path, cfg)
    Y_tmp = filters_rpn.createTargets(imageMeta, imageDims, cfg)
    y = filters_rpn.reduceData(Y_tmp, cfg)

    #STAGE 1
    proposals = Stages.stageone([X], y, imageMeta, imageDims, do_regr=True)
    
    [rois] = np.copy(proposals)
        
    # det prepare
    rois, target_props, target_deltas, IouS = filters_detection.createTargets(rois, imageMeta, imageDims, obj_mapping, cfg)
    img = np.copy(X[0])
    img = img + cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gtBox = draw.drawGTBoxes(img, imageMeta, imageDims)
    draw.drawAnchors(img, proposals[0], cfg)
    posAnc = draw.drawOverlapAnchors(img, proposals[0], imageMeta, imageDims, cfg)

    