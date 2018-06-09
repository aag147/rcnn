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

import numpy as np
import utils
import draw
import methods,\
       stages
import filters_detection,\
       filters_rpn,\
       filters_hoi
import cv2 as cv
import copy as cp

from keras.models import Sequential, Model

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
if True:
    # Load models
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=True)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')
    
    
genIterator = genTrain.begin()


for i in range(1):
    
    #################
    ##### train #####
    #################
    if False:
        [X, hbb, obb, ip], target_labels, imageMeta, imageDims, times = next(genIterator)
        h_bboxes, o_bboxes = filters_hoi.unprepareInputs(hbb, obb, imageDims)
        newh_bboxes, newo_bboxes = filters_hoi.prepareInputs(h_bboxes, o_bboxes, imageDims)
    
    
    #################
    ##### test ######
    #################
    X, y, imageMeta, imageDims, times = next(genIterator)  
    utils.update_progress_new(i+1, genTrain.nb_batches, imageMeta['id'])
    
    #STAGE 1
    proposals = Stages.stageone(X, y, imageMeta, imageDims)
    
    #STAGE 2
    bboxes = Stages.stagetwo(proposals, imageMeta, imageDims)
    if bboxes is None:
        print('Stage two error')
        continue
    
    #STAGE 3
    hbboxes, obboxes, props = Stages.stagethree(bboxes, imageMeta, imageDims)
    if hbboxes is None:
        print('Stage three error')
        continue
    
    #DRAW
    img = filters_rpn.unprepareInputs(X, imageDims, cfg)
    draw.drawGTBoxes(img, imageMeta, imageDims)
    draw.drawAnchors(img, proposals, cfg)
    draw.drawAnchors(img, bboxes, cfg)
    draw.drawPositiveHoIs(img, hbboxes, obboxes, props, hoi_mapping, imageMeta, imageDims, cfg)