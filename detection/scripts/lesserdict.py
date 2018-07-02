# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import numpy as np

import utils,\
       extract_data,\
       methods,\
       losses,\
       callbacks,\
       filters_helper as helper
from det_generators import DataGenerator
    
import filters_rpn
import filters_detection

if False:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    obj_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
#    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)


imageID = '487566'
imageMeta = genTrain.imagesMeta[imageID]
imageInputs = genTrain.imagesInputs[imageID]
X, imageDims = filters_rpn.prepareInputs(imageMeta, genTrain.images_path, cfg)

Y_tmp = filters_detection.loadData(imageInputs, cfg)


for i in range(1):
    bboxes1, target_labels, target_deltas = filters_detection.reduceData(Y_tmp, cfg)
    bboxes2 = np.copy(bboxes1)
    bboxes2 = filters_detection.prepareInputs(bboxes2, imageDims, imageMeta) 

import draw
import filters_detection

img = np.copy(X[0])
img += cfg.PIXEL_MEANS
img = img.astype(np.uint8)
bboxes2 = filters_detection.unprepareInputs(bboxes2, imageDims)

draw.drawOverlapAnchors(img, bboxes2[0], imageMeta, imageDims, cfg)
draw.drawGTBoxes(img, imageMeta, imageDims)

if False:
    redux = {}
    imageID = '487566'
    redux[imageID] = genTrain.imagesInputs[imageID]
    
    i = 0
    goal = 5000
    
    for imageID, inputMeta in genTrain.imagesInputs.items():
        redux[imageID] = inputMeta
        utils.update_progress_new(i+1, goal, imageID)
    
        if i == goal:
            break
        i += 1
    
    #utils.save_obj(redux, cfg.my_output_path + 'proposals_redux')