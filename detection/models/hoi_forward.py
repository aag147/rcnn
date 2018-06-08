# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:19:29 2018

@author: aag14
"""
import filters_helper as helper,\
       filters_rpn,\
       filters_detection,\
       filters_hoi
       
import numpy as np,\
       copy as cp,\
       math,\
       os

def forward_pass(X, models, cfg, class_mapping):
    [model_rpn, model_detection, model_hoi] = models
    
    #rpn prepare
    [X, y, imageMeta, imageDims] = X
    
    #rpn predict
    Y1_rpn, Y2_rpn, F = model_rpn.predict_on_batch(X)
    
    #rpn post
    pred_anchors = helper.deltas2Anchors(Y1_rpn, Y2_rpn, cfg, imageDims, do_regr=True)
    pred_anchors = helper.non_max_suppression_fast(pred_anchors, overlap_thresh=cfg.rpn_nms_overlap_thresh)
    pred_anchors = pred_anchors[:,0:4]
    
    # det prepare
    rois, target_props, target_deltas, IouS = filters_detection.prepareTargets(pred_anchors, imageMeta, imageDims, class_mapping, cfg)
    rois_norm = filters_detection.prepareInputs(rois, imageDims)    
    
    # det predict
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
    
    # det post
    rois = filters_detection.unprepareInputs(rois_norm, imageDims)
    boxes = helper.deltas2Boxes(allY1, allY2, rois, cfg)
    
    if len(boxes) == 0:
        return None, None, None
    
    boxes_nms = helper.non_max_suppression_boxes(boxes, cfg, cfg.det_nms_overlap_thresh_test)
    
    # hoi prepare
    h_bboxes, o_bboxes, hoi_labels, final_vals = filters_hoi.prepareTargets(boxes_nms, imageMeta, imageDims, cfg, class_mapping)
    
    if h_bboxes is None:
        return None, None, None
    
    patterns = filters_hoi.prepareInteractionPatterns(cp.copy(h_bboxes), cp.copy(o_bboxes), cfg)
    h_bboxes_norm, o_bboxes_norm = filters_hoi.prepareInputs(h_bboxes, o_bboxes, imageDims)
    
    nb_hoi_rois = h_bboxes.shape[0]
    
    # hoi predict
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
    hoi_labels = allhois[0]
    
    return h_bboxes, o_bboxes, hoi_labels