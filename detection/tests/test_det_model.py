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
       stages,\
       filters_detection,\
       filters_helper as helper
import utils
import draw
import numpy as np
import cv2 as cv

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
 
    
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)
    
#if True:
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')


genIterator = genVal.begin()

for i in range(1):
#    X, y, imageMeta, imageDims, times = next(genIterator)
#    imageMeta = genVal.imagesMeta['176847']
#    X, y, imageDims = Stages.stagezero(imageMeta, genVal.data_type)
    imageID = imageMeta['imageName'].split('.')[0]
    img = np.copy(X[0])
    img -= np.min(img)
    img /= np.max(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #STAGE 1
    print('Stage one...')
#    proposals = Stages.stageone([X], y, imageMeta, imageDims, do_regr=True)
    print('Stage two...')
#    rois, target_props, target_deltas, IouS = filters_detection.createTargets(proposals, imageMeta, imageDims, obj_mapping, cfg)
#    bboxes = helper.deltas2Boxes(target_props, target_deltas[:,:,80:], rois, imageDims, cfg)
#    bboxes = Stages.stagetwo([X,proposals], imageMeta, imageDims)
    
    print('Draw stuff...')
    draw.drawGTBoxes(img, imageMeta, imageDims)
    overlapAnchors = draw.drawOverlapAnchors(img, proposals[0], imageMeta, imageDims, cfg)
#    draw.drawPositiveRois(img, bboxes[0], obj_mapping)
    overlapRois = draw.drawOverlapRois(img, bboxes[0], imageMeta, imageDims, cfg, obj_mapping)