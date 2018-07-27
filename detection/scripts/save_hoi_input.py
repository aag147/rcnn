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
       filters_rpn,\
       filters_hoi
import hoi_test
import numpy as np
import utils

import os


if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels    
    
#     Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)

    sys.stdout.flush()

#if True:
#    cfg.det_nms_overlap_thresh = 0.5
#    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')
#    imageInputs, imageID, bboxes = hoi_test.saveInputData(genTest, Stages, cfg)
#    cfg.det_nms_overlap_thresh = 0.9
#    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')
#    imageInputs, imageID, bboxes = hoi_test.saveInputData(genTrain, Stages, cfg)
    cfg.det_nms_overlap_thresh = 0.5
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')
    imageInputs, imageID, bboxes = hoi_test.saveInputData(genTrain, Stages, cfg)
if False:
#    imageID = 'HICO_train2015_00025124'
    imageInputs = utils.load_obj(cfg.my_output_path + imageID)
#    keys = list(inputMeta.keys())
    
#    for imageID in keys:
    imageMeta = genTrain.imagesMeta[imageID]
#        imageInputs = inputMeta[imageID]

    X, imageDims = filters_rpn.prepareInputs(imageMeta, genTrain.images_path, cfg)
    Y_tmp = filters_hoi.loadData(imageInputs, imageDims, cfg)

    hbboxes, obboxes, target_labels, val_map = Y_tmp
    obboxescp = np.copy(obboxes)
    hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, cfg)
    patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, cfg)
    hcrops, ocrops = filters_hoi.convertBB2Crop(X, hbboxes, obboxes, imageDims)

    import draw
    img = np.copy(X[0])
    img += cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    draw.drawGTBoxes(img, imageMeta, imageDims)
#    draw.drawPositiveCropHoI(hbboxes[0], obboxes[0], hcrops, ocrops, patterns[0], target_labels[0], imageMeta, imageDims, cfg, obj_mapping)
    draw.drawPositiveHoI(img, hbboxes[0], obboxes[0], patterns[0], target_labels[0], imageMeta, imageDims, cfg, obj_mapping)
    draw.drawHumanAndObjectRois(img, bboxes[0], imageMeta, obj_mapping)

print()
print('Path:', cfg.my_output_path)


#[23 33 31 40  0 56]
#[23 31 32 45  0 56]
#[ 9 39 34 54  0 56]
#[27 40 33 46  0 56]
#[20 31 33 46  0 56]
#[26 41 33 47  0 56]
#[23 31 34 46  0 56]
#[24 32 32 40  0 56]
#[23 32 31 40  0 56]