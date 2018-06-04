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
import filters_helper as helper
import methods
import os
import filters_detection,\
       filters_rpn,\
       filters_hoi
import cv2 as cv
import copy as cp
import math

from keras.models import Sequential, Model
import h5py

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    class_mapping = data.class_mapping
    labels = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    
    if not os.path.exists(cfg.my_save_path):
        os.makedirs(cfg.my_save_path)
    print('results path', cfg.my_save_path)

if True:
    
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg, mode='train')
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg, mode='test')
    if type(cfg.my_weights)==str and len(cfg.my_weights)>0:
        print('Loading my weights...')
        
        path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
        model_rpn.load_weights(path, by_name=False)
        print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
        
        path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_detection.layers[6].get_weights()[0][0,0])
        model_detection.load_weights(path, by_name=True)
        print(model_detection.layers[6].get_weights()[0][0,0])

total_times = np.zeros((genTrain.nb_batches, 4))

genIterator = genTrain.begin()
detMeta = {}

for i in range(genTrain.nb_batches):
    X, y, imageMeta, imageDims, times = next(genIterator)
    imageID = imageMeta['id']
    
    
    s_rpn = time.time()
    Y1, Y2, F = model_rpn.predict_on_batch(X)
    f_rpn = time.time()
    
    pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims, do_regr=True)
    pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.rpn_nms_overlap_thresh)
    pred_anchors = pred_anchors[:,0:4]
    
    rois, target_props, target_deltas, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, data.class_mapping, cfg)
    rois_norm = filters_detection.prepareInputs(rois, imageDims)
    
    f_rpn_post = time.time()
    
    
    s_det = time.time()
    if True:
        allrois = np.zeros((1,cfg.rpn_nms_max_boxes, 5))
        allY1 = np.zeros((1,cfg.rpn_nms_max_boxes, cfg.nb_object_classes))
        allY2 = np.zeros((1,cfg.rpn_nms_max_boxes, (cfg.nb_object_classes-1)*4))
        for batchidx in range(math.ceil(cfg.rpn_nms_max_boxes / cfg.nb_detection_rois)):
            Y1, Y2 = model_detection.predict_on_batch([F, rois_norm])
            
            sidx = batchidx * cfg.nb_detection_rois
            fidx = min(cfg.rpn_nms_max_boxes, sidx + cfg.nb_detection_rois)
            allrois[:,sidx:fidx,:] = rois_norm[:,:fidx-sidx,:]
            allY1[:,sidx:fidx,:] = Y1[:,:fidx-sidx,:]
            allY2[:,sidx:fidx,:] = Y2[:,:fidx-sidx,:]
        
    f_det = time.time()
    rois = filters_detection.unprepareInputs(allrois, imageDims)
    boxes = helper.deltas2Boxes(allY1, allY2, rois, cfg)
    
    if len(boxes)==0:
        detMeta[imageID] = None
        continue
    
    boxes_nms = helper.non_max_suppression_boxes(boxes, cfg, cfg.det_nms_overlap_thresh)
    h_bboxes, o_bboxes, hoi_labels, val_map = filters_hoi.prepareTargets(boxes_nms, imageMeta, imageDims, cfg, class_mapping)

    if h_bboxes is None:
        detMeta[imageID] = None
        continue
    
    
    f_det_post = time.time()
    
    total_times[i, :] = [f_rpn-s_rpn, f_rpn_post-f_rpn, f_det-s_det, f_det_post-f_det]    
    
    h_bboxes_hoi, o_bboxes_hoi, hoi_labels_hoi, val_map_hoi = helper.bboxes2HOIformat(h_bboxes, o_bboxes, hoi_labels, val_map)
    detMeta[imageID] = {'imageName': imageMeta['imageName'], 'h_bboxes':h_bboxes_hoi, 'o_bboxes':o_bboxes_hoi, 'hoi_labels':hoi_labels_hoi, 'val_map':val_map_hoi}

    utils.update_progress_new(i+1, genTrain.nb_batches, imageMeta['id'])

path = cfg.my_save_path + 'hoiputs'
utils.save_dict(detMeta, path)
print()
print('Path:', cfg.my_save_path)
print('Times', np.mean(total_times, axis=0))
