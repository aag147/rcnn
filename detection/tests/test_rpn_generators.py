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
    
if False:
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        print('Loading my weights...')
        path = cfg.my_weights_path + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        model_rpn.load_weights(path) 

trainIterator = genTrain.begin()

total_times = np.array([0.0,0.0])

i = 0
for imageID, imageMeta in data.trainGTMeta.items():
    utils.update_progress_new(i+1, len(data.trainGTMeta), imageID)
    
    for obj in imageMeta['objects']:
        
        if obj['xmax'] <= obj['xmin']+1 or obj['ymax'] <= obj['ymin']+1:
            print(imageID)
            print(obj)
    i += 1

#i = 0
#imageMeta = imagesMeta['200365']
##
##newImageMeta = {'imageName': imageMeta['imageName'], 'objects': []}
##for obj in imageMeta['objects']:
##    if obj['label'] == 'hot dog':
##        if obj['ymax'] - obj['ymin'] < 30:
##            continue
##    newImageMeta['objects'].append(obj)
#        
#import filters_rpn
#img, imageDims = filters_rpn.prepareInputs(imageMeta, cfg.data_path + 'images/train/', cfg)
#draw.drawGTBoxes((img[0]+1.0)/2.0, imageMeta, imageDims)
#
##utils.save_dict(imagesMeta, cfg.data_path + 'imagesMeta')

j = 0
for i in range(0):
    if False:
        X, y, imageMeta, imageDims, times = next(trainIterator)
#        X, y, imageMeta, imageDims, times = next(trainIterator)
    if False:
        img = cp.copy(X[0])
        img += 1.0
        img /= 2.0

    if False:
        Y1 = y[0][:,:,:,12:];
        idxs = y[0][:,:,:,:12]; idxs = idxs.reshape((-1)); idxs = np.where(idxs)[0]
        
        Y2 = y[1][:,:,:,48:]
                
        props = np.reshape(Y1,(-1,1))
    
    Y1, Y2 = model_rpn.predict_on_batch(X)
    all_pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims, do_regr=True)
#    pred_anchors = pred_anchors[idxs,:]
    print('positives', np.sum(Y1>0.5))
    print(all_pred_anchors.shape)
#    draw.drawPositiveAnchors(img, pred_anchors, cfg)
    t_start = time.time()
    pred_anchors = helper.non_max_suppression_fast(all_pred_anchors, overlap_thresh=cfg.detection_nms_overlap_thresh)
    t_end = time.time()
    print('time', t_end-t_start)
    print('positives', np.sum(pred_anchors[:,4]>0.5))
    print(pred_anchors.shape)
    draw.drawAnchors(img, pred_anchors, cfg)
    draw.drawGTBoxes(img, imageMeta, imageDims)
    
#print(f-s)

