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
    genTrain = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
if False:
#    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg, mode='train')
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg, mode='test')
    if type(cfg.my_weights)==str and len(cfg.my_weights)>0:
        print('Loading my weights...')
        path = cfg.my_weights_path + cfg.my_weights
        path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        print(path)
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_rpn.layers[19].get_weights()[0][0,0,0,0])
        model_rpn.load_weights(path, by_name=False)
        print(model_rpn.layers[19].get_weights()[0][0,0,0,0])
       
        path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        print(path)
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_detection.layers[4].get_weights()[0][0,0])
        model_detection.load_weights(path, by_name=True)
        print(model_detection.layers[4].get_weights()[0][0,0])
        
        

#model_rpn.save_weights(cfg.part_results_path + 'rpn_model.h5')
#model_detection.save_weights(cfg.part_results_path + 'det_model.h5')
        
#shared_cnn = Model(model_detection.input, model_detection.layers[17].output)
#shared_cnn.save(cfg.part_results_path + 'shared_model.h5')
#
#print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
#model_rpn.load_weights(cfg.part_results_path + 'shared_model.h5', by_name=True) 
#print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
#
#trainIterator = genTrain.begin()

total_times = np.array([0.0,0.0])
j = 0

#images_path = cfg.data_path + 'images/'
#images_path = images_path + 'train/'
#rois_path = cfg.my_detections_path
genIterator = genTrain.begin()



#final_hbbs, final_obbs, final_labels, final_vals = filters_hoi.prepareTargets(boxes_nms, imageMeta, imageDims, cfg, class_mapping)

for i in range(0):
    X, y, imageMeta, imageDims, times = next(genIterator)
    print(imageMeta['imageName'])

#    path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir + '/weights/' + cfg.my_weights
#    assert os.path.exists(path), 'invalid path: %s' % path
#    print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
#    model_rpn.load_weights(path, by_name=False)
#    print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
    
    Y1, Y2, F = model_rpn.predict_on_batch(X)
    
    pred_anchors = helper.deltas2Anchors(Y1, Y2, cfg, imageDims, do_regr=True)
    pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.rpn_nms_overlap_thresh)
    pred_anchors = pred_anchors[:,0:4]
    
    rois, target_props, target_deltas, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, data.class_mapping, cfg)
    rois_norm = filters_detection.prepareInputs(rois, imageDims)
    
    
#    path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir + '/weights/' + cfg.my_weights
#    assert os.path.exists(path), 'invalid path: %s' % path
##    print(model_detection.layers[11].get_weights()[0][0,0,0,0])
#    model_detection.load_weights(path, by_name=True)
##    print(model_detection.layers[11].get_weights()[0][0,0,0,0])
    
    
    if True:
        allY1 = np.zeros((1,cfg.rpn_nms_max_boxes, cfg.nb_object_classes))
        allY2 = np.zeros((1,cfg.rpn_nms_max_boxes, (cfg.nb_object_classes-1)*4))
        for batchidx in range(math.ceil(cfg.rpn_nms_max_boxes / cfg.nb_detection_rois)):
            sidx = batchidx * cfg.nb_detection_rois
            fidx = min(cfg.rpn_nms_max_boxes, sidx + cfg.nb_detection_rois)
            batch_rois = rois_norm[:,sidx:fidx,:]
            Y1, Y2 = model_detection.predict_on_batch([F, batch_rois])

            allY1[:,sidx:fidx,:] = Y1[:,:fidx-sidx,:]
            allY2[:,sidx:fidx,:] = Y2[:,:fidx-sidx,:]

    draw.drawGTBoxes((X[0]+1.0)/2.0, imageMeta, imageDims)
    
    rois = filters_detection.unprepareInputs(rois_norm, imageDims)
    boxes = helper.deltas2Boxes(allY1, allY2, rois, cfg)
    draw.drawAnchors((X[0]+1.0)/2.0, boxes, cfg)
    
    boxes_nms = helper.non_max_suppression_boxes(boxes, cfg, 0.5)
    draw.drawAnchors((X[0]+1.0)/2.0, boxes_nms, cfg)
    
    if False:
        h_bboxes, o_bboxes, hoi_labels, final_vals = filters_hoi.prepareTargets(boxes_nms, imageMeta, imageDims, cfg, class_mapping)
    #    h_bboxes_norm, o_bboxes_norm = filters_hoi.prepareInputs(h_bboxes, o_bboxes, imageDims)
        h_bboxes_hoi, o_bboxes_hoi, hoi_labels_hoi, val_map_hoi = helper.bboxes2HOIformat(h_bboxes, o_bboxes, hoi_labels, final_vals)
        patterns = filters_hoi.prepareInteractionPatterns(h_bboxes, o_bboxes, cfg)
        for hoiidx in range(10):
            draw.drawHoIComplete((X[0]+1.0)/2.0, h_bboxes[hoiidx,:], o_bboxes[hoiidx,:], patterns[0,hoiidx,::], hoi_labels[hoiidx,:], labels)
        break


