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


class AllStages:
    def __init__(self, cfg, Models, obj_mapping, hoi_mapping, mode='train'):
        self.mode = mode
        self.cfg = cfg
        self.obj_mapping = obj_mapping
        self.hoi_mapping = hoi_mapping
        self.model_rpn, self.model_det, self.model_hoi = Models.get_models()
        
        self.images_path = cfg.data_path + 'images/'
        self.anchors_path = cfg.data_path + 'anchors/'


    def stagezero(self, imageMeta, data_type):
        images_path = self.images_path + data_type + '/'
        anchors_path = self.anchors_path + data_type + '/'
        X, imageDims = filters_rpn.prepareInputs(imageMeta, images_path, self.cfg)
        y = None        
        if self.mode=='train':
            y_tmp = filters_rpn.loadData(imageMeta, anchors_path, self.cfg)
            if y_tmp is None:
                y_tmp = filters_rpn.createTargets(imageMeta, imageDims, self.cfg)
            y = filters_rpn.reduceData(y_tmp, self.cfg)
        return X, y, imageDims

    def stageone(self, X, y, imageMeta, imageDims, include='all', do_regr = True):        
        #rpn prepare
        img = X
                
        #rpn predict
        if self.mode == 'test' and self.cfg.use_shared_cnn:
            rpn_props, rpn_deltas, F = self.model_rpn.predict_on_batch(img)
            self.shared_cnn = F
        elif self.mode == 'test':
            rpn_props, rpn_deltas = self.model_rpn.predict_on_batch(img)
            self.shared_cnn = img
        self.shared_img = img
        
        #rpn post
        proposals = helper.deltas2Anchors(rpn_props, rpn_deltas, self.cfg, imageDims, do_regr=do_regr)
        proposals = helper.non_max_suppression_fast(proposals, overlap_thresh = self.cfg.rpn_nms_overlap_thresh_test)
        return proposals
    
    def stagetwo(self, X, imageMeta, imageDims, include='all'):
        # det prepare
        proposals = X
        if self.mode == 'test' and include != 'pre':
            rois = np.expand_dims(proposals, axis=0)
        else:
            rois, target_props, target_deltas, IouS = filters_detection.createTargets(proposals, imageMeta, imageDims, self.obj_mapping, self.cfg)
            if include=='pre':
                return rois, target_props, target_deltas
        
        rois_norm = filters_detection.prepareInputs(rois, imageDims)        
        
        # det predict
        nb_det_rois = rois.shape[1]
        all_det_props = np.zeros((1,nb_det_rois, self.cfg.nb_object_classes))
        all_det_deltas = np.zeros((1,nb_det_rois, (self.cfg.nb_object_classes-1)*4))
        for batchidx in range(math.ceil(nb_det_rois / self.cfg.nb_detection_rois)):    
            sidx = batchidx * self.cfg.nb_detection_rois
            fidx = min(nb_det_rois, sidx + self.cfg.nb_detection_rois)
            batch_rois = rois_norm[:,sidx:fidx,:]
            det_props, det_deltas = self.model_det.predict_on_batch([self.shared_cnn, batch_rois])

            all_det_props[:,sidx:fidx,:] = det_props[:,:fidx-sidx,:]
            all_det_deltas[:,sidx:fidx,:] = det_deltas[:,:fidx-sidx,:]
        
        # det post
        rois = filters_detection.unprepareInputs(rois_norm, imageDims)
        bboxes = helper.deltas2Boxes(all_det_props, all_det_deltas, rois, imageDims, self.cfg)
#        bboxes = helper.deltas2ObjBoxes(all_det_props, all_det_deltas, rois, imageDims, self.cfg, self.obj_mapping)
        
        if len(bboxes) == 0:
            return None
        bboxes_nms = helper.non_max_suppression_boxes(bboxes, self.cfg, self.cfg.det_nms_overlap_thresh_test)
#        bboxes_nms = helper.non_max_suppression_fast(bboxes, self.cfg.det_nms_overlap_thresh_test)
        
        return bboxes_nms
    
    def stagethree(self, X, imageMeta, imageDims, include='all'):
        # hoi prepare
        bboxes = X
        
        if self.mode == 'test' and include != 'pre':
            all_hbboxes, all_obboxes = filters_hoi.splitInputs(bboxes)
        else:
            all_hbboxes, all_obboxes, all_target_labels, val_map = filters_hoi.createTargets(bboxes, imageMeta, imageDims, self.cfg, self.obj_mapping)
            if include=='pre':
                return all_hbboxes, all_obboxes, all_target_labels, val_map
        
        if all_hbboxes is None:
            return None, None, None
        
        patterns = filters_hoi.createInteractionPatterns(all_hbboxes, all_obboxes, self.cfg)
        hbboxes_norm, obboxes_norm = filters_hoi.prepareInputs(all_hbboxes, all_obboxes, imageDims)
                
        # hoi predict
        nb_hoi_rois = hbboxes_norm.shape[1]
        all_hoi_props = np.zeros((1,nb_hoi_rois, self.cfg.nb_hoi_classes))
        all_hoi_hbboxes = np.zeros((1,nb_hoi_rois, 5))
        all_hoi_obboxes = np.zeros((1,nb_hoi_rois, 5))
        for batchidx in range(math.ceil(nb_hoi_rois / self.cfg.nb_hoi_rois)):    
            sidx = batchidx * self.cfg.nb_hoi_rois
            fidx = min(nb_hoi_rois, sidx + self.cfg.nb_hoi_rois)
            batch_h = hbboxes_norm[:,sidx:fidx,:]
            batch_o = obboxes_norm[:,sidx:fidx,:]
            batch_p = patterns[:,sidx:fidx,:,:,:]
            
            hoi_props, hoi_hbboxes, hoi_obboxes = self.model_hoi.predict_on_batch([self.shared_img, batch_h, batch_o, batch_p])

            all_hoi_props[:,sidx:fidx,:] = hoi_props[:,:fidx-sidx,:]
            all_hoi_hbboxes[:,sidx:fidx,:] = hoi_hbboxes[:,:fidx-sidx,:]
            all_hoi_obboxes[:,sidx:fidx,:] = hoi_obboxes[:,:fidx-sidx,:]
        
        # hoi post
        all_hoi_hbboxes, all_hoi_obboxes = filters_hoi.unprepareInputs(all_hoi_hbboxes, all_hoi_obboxes, imageDims)
        return all_hoi_hbboxes[0], all_hoi_obboxes[0], all_hoi_props[0]