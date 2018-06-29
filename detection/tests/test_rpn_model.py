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
#    proposals = Stages.stageone([X], y, imageMeta, imageDims, do_regr=True)
    
    [rois] = np.copy(proposals)
        
    # det prepare
    rois, target_props, target_deltas, IouS = filters_detection.createTargets(rois, imageMeta, imageDims, obj_mapping, cfg)
#    rois_norm = filters_detection.prepareInputs(rois, imageDims)        
    
    # det post
#    rois1 = np.copy(rois)
#    rois = filters_detection.unprepareInputs(rois_norm, imageDims)
    bboxes1 = helper.deltas2Boxes(np.copy(target_props), np.copy(target_deltas), np.copy(rois), imageDims, cfg)
#    bboxes_nms1 = helper.non_max_suppression_boxes(bboxes1, cfg, cfg.det_nms_overlap_thresh)
#    bboxes_nms1 = np.expand_dims(bboxes_nms1, axis=0)
    
    bboxes2 = helper.deltas2ObjBoxes(np.copy(target_props), np.copy(target_deltas), np.copy(rois), imageDims, cfg, obj_mapping)
#    bboxes_nms2 = helper.non_max_suppression_boxes(bboxes2, cfg, cfg.det_nms_overlap_thresh)
#    bboxes_nms2 = np.expand_dims(bboxes_nms2, axis=0)
    
    img = np.copy(X[0])
    img = img + cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    gtBox = draw.drawGTBoxes(img, imageMeta, imageDims)
    draw.drawAnchors(img, proposals[0], cfg)
    draw.drawAnchors(img, bboxes_nms1[0], cfg)
    draw.drawAnchors(img, bboxes_nms2[0], cfg)
    posAnc = draw.drawOverlapAnchors(img, proposals[0], imageMeta, imageDims, cfg)
    posAnc = draw.drawOverlapAnchors(img, bboxes1, imageMeta, imageDims, cfg)
    posAnc = draw.drawOverlapAnchors(img, bboxes2, imageMeta, imageDims, cfg)
    