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
import time
import metrics
import filters_helper as helper
import methods
import os
import filters_detection,\
       filters_rpn,\
       filters_hoi,\
       stages
import cv2 as cv
import copy as cp
import math

from keras.models import Sequential, Model
import h5py
#import draw


np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
#if True:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    
#if True:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=True)
#if True:
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')
    
total_times = np.array([0.0,0.0])
genIterator = genTrain.begin()

coco_res = []
for i in range(genTrain.nb_batches):

    X, y, imageMeta, imageDims, times = next(genIterator)
    img = filters_rpn.unprepareInputs(X, imageDims, cfg)
#    draw.drawGTBoxes(img, imageMeta, imageDims)
    
    utils.update_progress_new(i+1, genTrain.nb_batches, imageMeta['id'])
    
    #STAGE 1
    proposals = Stages.stageone(X, y, imageMeta, imageDims)
#    draw.drawAnchors(img, proposals, cfg)
    
    #STAGE 2
    bboxes = Stages.stagetwo(proposals, imageMeta, imageDims)
    if bboxes is None:
        print('Stage two error')
        continue
#    draw.drawAnchors(img, bboxes, cfg)
    
    #STAGE 3
    hbboxes, obboxes, props = Stages.stagethree(bboxes, imageMeta, imageDims)
    if hbboxes is None:
        print('Stage three error')
        continue
#    draw.drawPositiveHoIs(img, hbboxes, obboxes, props, hoi_mapping, imageMeta, imageDims, cfg)
    
    #DONE
    cocoformat = filters_hoi.convertResults(hbboxes, obboxes, props, imageMeta, imageDims['scale'], cfg.rpn_stride)
    coco_res += cocoformat
    

if False:
    path = cfg.part_results_path + 'HICO/hoi5c/hoi_output'
    utils.save_dict(coco_res, path)
    
    mAP, AP = metrics.computeHOImAP(coco_res, data.trainGTMeta, obj_mapping, hoi_mapping, cfg)
    saveMeta = {'mAP': mAP, 'AP': AP.tolist()}
    path = cfg.part_results_path + 'HICO/hoi5c/hoimap'
    utils.save_dict(saveMeta, path)
    
    print('mAP', mAP)