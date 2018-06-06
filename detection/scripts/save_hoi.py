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
import metrics
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
    data = extract_data.object_data(False)
    cfg = data.cfg
    class_mapping = data.class_mapping
    labels = data.hoi_labels
    
if True:
    # Create batch generators
    imagesMeta = data.testGTMeta
    genTrain = DataGenerator(imagesMeta = imagesMeta, cfg=cfg, data_type='test', do_meta=True)
    
if True:
#    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg, mode='train')
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
        print(model_detection.layers[4].get_weights()[0][0,0])
        model_detection.load_weights(path, by_name=True)
        print(model_detection.layers[4].get_weights()[0][0,0])

        print(model_rpn.layers[11].get_weights()[0][0,0,0,0])

total_times = np.array([0.0,0.0])
j = 0

genIterator = genTrain.begin()

coco_res = []
for i in range(1):

    if True:
        # rpn
        X, y, imageMeta, imageDims, times = next(genIterator)

        Y1_rpn, Y2_rpn, F = model_rpn.predict_on_batch(X)
        draw.drawGTBoxes((X[0]+1.0)/2.0, imageMeta, imageDims)
        
        #rpn post
        pred_anchors = helper.deltas2Anchors(Y1_rpn, Y2_rpn, cfg, imageDims, do_regr=True)
        print(pred_anchors.shape)
        pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.rpn_nms_overlap_thresh)
        print(pred_anchors.shape)
        pred_anchors = pred_anchors[:,0:4]
        
        # detection
        rois, target_props, target_deltas, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, data.class_mapping, cfg)
        rois_norm = filters_detection.prepareInputs(rois, imageDims)    
        
        nb_det_rois = rois.shape[0]
        if True:
            allY1 = np.zeros((1,nb_det_rois, cfg.nb_object_classes))
            allY2 = np.zeros((1,nb_det_rois, (cfg.nb_object_classes-1)*4))
            for batchidx in range(math.ceil(nb_det_rois / cfg.nb_detection_rois)):    
                sidx = batchidx * cfg.nb_detection_rois
                fidx = min(nb_det_rois, sidx + cfg.nb_detection_rois)
                batch_rois = rois_norm[:,sidx:fidx,:]
                Y1, Y2 = model_detection.predict_on_batch([F, batch_rois])
    
                allY1[:,sidx:fidx,:] = Y1[:,:fidx-sidx,:]
                allY2[:,sidx:fidx,:] = Y2[:,:fidx-sidx,:]
        
        # detection post
        rois = filters_detection.unprepareInputs(rois_norm, imageDims)
        boxes = helper.deltas2Boxes(allY1, allY2, rois, cfg)
        boxes_nms = helper.non_max_suppression_boxes(boxes, cfg, 0.5)
        
        
        # hoi
        path = cfg.part_results_path + 'HICO/hoi5cf/weights/' + cfg.my_weights
        print(path)
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_hoi.layers[23].get_weights()[0][0,0,0,0])
        model_hoi.load_weights(path, by_name=False)
        print(model_hoi.layers[23].get_weights()[0][0,0,0,0])
        
        
        h_bboxes, o_bboxes, hoi_labels, final_vals = filters_hoi.prepareTargets(boxes_nms, imageMeta, imageDims, cfg, class_mapping)
        patterns = filters_hoi.prepareInteractionPatterns(cp.copy(h_bboxes), cp.copy(o_bboxes), cfg)
        h_bboxes_norm, o_bboxes_norm = filters_hoi.prepareInputs(h_bboxes, o_bboxes, imageDims)
        
        nb_hoi_rois = h_bboxes.shape[0]
        
        if True:
            allhois = np.zeros((1,nb_hoi_rois, cfg.nb_hoi_classes))
            allhbbs = np.zeros((1,nb_hoi_rois, 5))
            allobbs = np.zeros((1,nb_hoi_rois, 5))
            for batchidx in range(math.ceil(nb_hoi_rois / cfg.nb_hoi_rois)):    
                sidx = batchidx * cfg.nb_hoi_rois
                fidx = min(nb_hoi_rois, sidx + cfg.nb_hoi_rois)
                batch_h = h_bboxes_norm[:,sidx:fidx,:]
                batch_o = o_bboxes_norm[:,sidx:fidx,:]
                batch_p = patterns[:,sidx:fidx,:]
                pred_hoi, pred_hbb, pred_obb = model_hoi.predict_on_batch([X, batch_h, batch_o, batch_p])
    
                allhois[:,sidx:fidx,:] = pred_hoi[:,:fidx-sidx,:]
                allhbbs[:,sidx:fidx,:] = pred_hbb[:,:fidx-sidx,:]
                allobbs[:,sidx:fidx,:] = pred_obb[:,:fidx-sidx,:]
        
        # hoi post
        h_bboxes, o_bboxes = filters_hoi.unprepareInputs(allhbbs, allobbs, imageDims)        

    cocoformat = helper.hoiBBoxes2COCOformat(h_bboxes, o_bboxes, allhois, imageMeta, imageDims['scale'], cfg.rpn_stride)
    coco_res += cocoformat
    
    utils.update_progress_new(i+1, genTrain.nb_batches, imageMeta['id'])

mAP = metrics.computeHOImAP(coco_res, imagesMeta, class_mapping, labels, cfg)