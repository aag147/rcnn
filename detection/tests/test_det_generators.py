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
import filters_detection,\
       filters_rpn
import cv2 as cv
import copy as cp
import math

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    cfg.faster_rcnn_config()
    # Create batch generators
#    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    
if True:
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg)
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        print('Loading my weights...')
        path = cfg.my_weights_path + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        model_detection.load_weights(path) 

#trainIterator = genTrain.begin()

total_times = np.array([0.0,0.0])
j = 0

imageMeta = data.trainGTMeta['HICO_train2015_00003244']
images_path = cfg.data_path + 'images/'
images_path = images_path + 'train/'
rois_path = cfg.my_detections_path

for i in range(1):
    img, imageDims = filters_rpn.prepareInputs(imageMeta, images_path, cfg)
    if False:
        allrois = np.zeros((1,cfg.detection_nms_max_boxes, 5))
        allY1 = np.zeros((1,cfg.detection_nms_max_boxes, cfg.nb_object_classes))
        allY2 = np.zeros((1,cfg.detection_nms_max_boxes, (cfg.nb_object_classes-1)*4))
        for batchidx in range(math.ceil(cfg.detection_nms_max_boxes / cfg.nb_detection_rois)):
            rois_norm, target_props, target_deltas = filters_detection.loadData(imageMeta, rois_path, cfg, batchidx)
            Y1, Y2 = model_detection.predict_on_batch([img, rois_norm])
            
            sidx = batchidx * cfg.nb_detection_rois
            fidx = min(cfg.detection_nms_max_boxes, sidx + cfg.nb_detection_rois)
            allrois[:,sidx:fidx,:] = rois_norm[:,:fidx-sidx,:]
            allY1[:,sidx:fidx,:] = Y1[:,:fidx-sidx,:]
            allY2[:,sidx:fidx,:] = Y2[:,:fidx-sidx,:]
            
    #        X, y, imageMeta, imageDims, times = next(trainIterator)
    #        break
    #        total_times += times
    #        utils.save_obj(y, cfg.data_path +'anchors/train/' + imageMeta['imageName'].split('.')[0])
    #        utils.update_progress_new(i, genTrain.nb_batches, list(times) + [0,0], imageMeta['imageName'])
        
    #    Y1, Y2 = model_detection.predict_on_batch([img, rois_norm])
    
    img = cp.copy(img[0])
    img += 1.0
    img /= 2.0
    
    draw.drawGTBoxes(img, imageMeta, imageDims)
    
    rois = filters_detection.unprepareInputs(allrois, imageDims)
    boxes = helper.deltas2Boxes(allY1, allY2, rois, cfg)
    draw.drawAnchors(img, boxes, cfg)
    
    boxes_nms = helper.non_max_suppression_boxes(boxes, cfg)
    draw.drawAnchors(img, boxes_nms[0:5], cfg)
    break


