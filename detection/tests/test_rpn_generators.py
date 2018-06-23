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
import draw
import filters_helper as helper
import methods
import os
import filters_detection
import filters_rpn
import cv2 as cv
import copy as cp

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    class_mapping = data.class_mapping
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')

trainIterator = genTrain.begin()

total_times = np.array([0.0,0.0])

data_type = 'train'
images_path = cfg.data_path + 'images/'
images_path = images_path + data_type + '/'

j = 0
for i in range(1):
    if True:
        X, y, imageMeta, imageDims, times = next(trainIterator)
#        imageMeta = data.valGTMeta['176847']
#        imageMeta = data.trainGTMeta['463564']
        X, imageDims = filters_rpn.prepareInputs(imageMeta, images_path, cfg)
        y = filters_rpn.createTargets(imageMeta, imageDims, cfg)

        Y1 = y[0][:,:,:,:]        
        Y2 = y[1][:,:,:,:]
                
        props = np.reshape(Y1,(-1,1))
    
    all_pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims, do_regr=False)
    print('positives', np.sum(Y1>0.5))
    print(all_pred_anchors.shape)
    img = np.copy(X[0]) + cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    draw.drawPositiveAnchors(img, all_pred_anchors, cfg)
    
    
#    t_start = time.time()
#    pred_anchors = helper.non_max_suppression_fast(all_pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh_test)
#    t_end = time.time()
#    print('time', t_end-t_start)
#    print('positives', np.sum(pred_anchors[:,4]>0.5))
#    print(pred_anchors.shape)
#    draw.drawAnchors((X[0]+1.0)/2.0, pred_anchors, cfg)
    draw.drawGTBoxes(img, imageMeta, imageDims)
    
#print(f-s)

