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
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)
#    genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=False, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')


genIterator = genVal.begin()

for i in range(1):
    X, y, imageMeta, imageDims, times = next(genIterator)
    imageID = imageMeta['imageName'].split('.')[0]
    
    #STAGE 1
    proposals = Stages.stageone([X], y, imageMeta, imageDims)
    
    img = np.copy(X[0]) + cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    gtBox = draw.drawGTBoxes(img, imageMeta, imageDims)
    posAnc = draw.drawOverlapAnchors(img, proposals[0], imageMeta, imageDims, cfg)