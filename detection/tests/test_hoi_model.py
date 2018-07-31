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
import filters_detection,\
       filters_helper as helper

import methods,\
       stages,\
       filters_rpn,\
       filters_hoi
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
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True, mode='test', approach='evalnew')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test', approach='new')
#    genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test', do_meta=True)
    

if True:
#    cfg.do_fast_hoi = True
    Models = methods.AllModels(cfg, mode='test', do_rpn=False, do_det=False, do_hoi=True)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')

iterator = genVal
genIterator = iterator.begin()

for i in range(1):
    [X, all_hbboxes, all_obboxes, all_val_map], all_target_labels, imageMeta, imageDims, _ = next(genIterator)
    imageID = imageMeta['imageName'].split('.')[0]

    print('Stage one...')
#    proposals = Stages.stageone([X], y, imageMeta, imageDims)
    print('Stage two...')
#    bboxes = Stages.stagetwo([X,proposals], imageMeta, imageDims)
    print('Stage three...')
#    all_hbboxes, all_obboxes, all_target_labels, all_val_map = Stages.stagethree_targets(bboxes, imageMeta, imageDims)
#    hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets([all_hbboxes, all_obboxes, all_target_labels, all_val_map], cfg)
#    all_hoi_hbboxes, all_hoi_obboxes, all_hoi_props = Stages.stagethree([X,bboxes], imageMeta, imageDims, obj_mapping, include='all')
    pred_hbboxes, pred_obboxes, pred_props = Stages.stagethree([X,all_hbboxes,all_obboxes], imageMeta, imageDims, obj_mapping)
#    batch_hcrop, batch_ocrop, batch_p, batch_h, batch_o = Stages.stagethree([X,all_hbboxes,all_obboxes], imageMeta, imageDims, obj_mapping)
    
    
    print('Draw...')
    img = np.copy(X[0])
    img += cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    bboxes = np.concatenate([all_hbboxes,all_obboxes],axis=1)[0][:,:]
    draw.drawGTBoxes(img, imageMeta, imageDims)
    draw.drawHoIExample(imageMeta, iterator.images_path, hoi_mapping)
    print('GT hoi labels')
    print('GT',np.unique(np.where(all_target_labels[0,:,:]>0)[0]), np.unique(np.where(all_target_labels[0,:,:]>0)[1]))
#    draw.drawPositiveCropHoI(batch_h[:,1:], batch_o[:,1:], batch_hcrop[:,1:], batch_ocrop[:,1:], batch_p, None, imageMeta, imageDims, cfg, obj_mapping)
#    draw.drawOverlapRois(img, bboxes, imageMeta, imageDims, cfg, obj_mapping)
    draw.drawPositiveRois(img, bboxes, obj_mapping)
    draw.drawOverlapAnchors(img, bboxes, imageMeta, imageDims, cfg)
#    idxs = draw.drawPositiveHoI(img, pred_hbboxes, pred_obboxes, None, pred_props, imageMeta, imageDims, cfg, obj_mapping)
    draw.drawOverlapHoI(img, pred_hbboxes, pred_obboxes, pred_props, imageMeta, imageDims, cfg, obj_mapping, hoi_mapping)
#    good_bboxes = np.copy(all_obboxes[0,idxs,:])
#    good_bboxes[:,2] += good_bboxes[:,0]
#    good_bboxes[:,3] += good_bboxes[:,1]