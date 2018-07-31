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
       filters_rpn,\
       filters_hoi
import utils
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
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test')
#    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test')
    

#if False:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=True)
Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test', return_times = True)

generator = genVal
genIterator = generator.begin()

nb_iterations = 1
all_times = np.zeros((nb_iterations, 5))

for i in range(nb_iterations):
    X, y, imageMeta, imageDims, times = next(genIterator)
    
#    imageID = 'Play_Saxophone_007'
#    imageMeta = generator.imagesMeta[imageID]
#    X, y, imageDims = Stages.stagezero(imageMeta, generator.data_type)
    imageID = imageMeta['imageName'].split('.')[0]
    
    if (i+1) % 100 == 0:
        utils.update_progress_new((i+1), nb_iterations, imageID)
    print('imageID', imageID)
    print('Stage one...')
    proposals, times = Stages.stageone([X], y, imageMeta, imageDims)
    all_times[i, 0:2] = times
    print('Stage two...')
    bboxes, times = Stages.stagetwo([proposals], imageMeta, imageDims)
    all_times[i, 2:4] = times
    print('Stage three...')
    pred_hbboxes, pred_obboxes, pred_props, times = Stages.stagethree([bboxes], imageMeta, imageDims, obj_mapping)
    all_times[i, 4:5] = times
    
#    continue
    
    gt_obj_label = obj_mapping[hoi_mapping[imageMeta['label']]['obj']]
    idxs = np.where((bboxes[0,:,5]==gt_obj_label) | (bboxes[0,:,5]==1))[0]
    bboxes_rdx = bboxes[:,idxs,:]
    
    
    import draw
    print('Draw...')
    img = np.copy(X[0])
    img += cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#    draw.drawGTBoxes(img, imageMeta, imageDims)
#    draw.drawHoIExample(imageMeta, generator.images_path, hoi_mapping)
#    print('GT hoi labels')
#    print('GT',np.unique(np.where(all_target_labels[0,:,:]>0)[0]), np.unique(np.where(all_target_labels[0,:,:]>0)[1]))
#    draw.drawPositiveCropHoI(batch_h[:,1:], batch_o[:,1:], batch_hcrop[:,1:], batch_ocrop[:,1:], batch_p, None, imageMeta, imageDims, cfg, obj_mapping)
#    draw.drawOverlapRois(img, bboxes, imageMeta, imageDims, cfg, obj_mapping)
    draw.drawPositiveRois(img, bboxes_rdx[0], obj_mapping)
#    draw.drawOverlapRois(img, bboxes[0], imageMeta, imageDims, cfg, obj_mapping)
    draw.drawPositiveHoI(img, pred_hbboxes, pred_obboxes, None, pred_props, imageMeta, imageDims, cfg, obj_mapping)
#    draw.drawOverlapHoI(img, pred_hbboxes, pred_obboxes, pred_props, imageMeta, imageDims, cfg, obj_mapping, hoi_mapping)

all_times_mean = np.mean(all_times, axis=0)
print(all_times_mean)