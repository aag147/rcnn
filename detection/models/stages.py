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
import time

class AllStages:
    def __init__(self, cfg, Models, obj_mapping, hoi_mapping, mode='train', return_times = False):
        self.mode = mode
        self.cfg = cfg
        self.obj_mapping = obj_mapping
        self.hoi_mapping = hoi_mapping
        self.return_times = return_times
        if Models is not None:
            self.model_rpn, self.model_det, self.model_hoi = Models.get_models()
        
        self.images_path = cfg.data_path + 'images/'
        self.anchors_path = cfg.data_path + 'anchors/'
        
        self.rpn_nms_thresh = self.cfg.rpn_nms_overlap_thresh if self.mode == 'train' else self.cfg.rpn_nms_overlap_thresh_test
        self.det_nms_thresh = self.cfg.det_nms_overlap_thresh if self.mode == 'train' else self.cfg.det_nms_overlap_thresh_test

        print('   RPN NMS thresh:', self.rpn_nms_thresh)
        print('   DET NMS thresh:', self.det_nms_thresh)

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
        [img] = X
        
        #rpn predict
        start_pred = time.time()
        if self.cfg.use_shared_cnn:
            rpn_props, rpn_deltas, F = self.model_rpn.predict_on_batch(img)
            self.shared_cnn = F
        else:
            rpn_props, rpn_deltas = self.model_rpn.predict_on_batch(img)
            self.shared_cnn = img
        self.shared_img = img
        end_pred = time.time()
        
        #rpn post
        start_pp = time.time()
        proposals = helper.deltas2Anchors(rpn_props, rpn_deltas, self.cfg, imageDims, do_regr=do_regr)
        proposals = helper.non_max_suppression_fast(proposals, overlap_thresh = self.rpn_nms_thresh)
        proposals = np.expand_dims(proposals, axis=0)
        end_pp = time.time()
        
        if self.return_times:
            return proposals, [end_pred-start_pred, end_pp-start_pp]
        else:
            return proposals
    
    def stagetwo_targets(self, X, imageMeta, imageDims):
        proposals = X[0,::]
        rois, target_props, target_deltas, IouS = filters_detection.createTargets(proposals, imageMeta, imageDims, self.obj_mapping, self.cfg)
        return rois, target_props, target_deltas
    
    def stagetwo(self, X, imageMeta, imageDims, include='all'):
        if len(X)==2:
            [self.shared_cnn, rois] = X
        else:
            [rois] = X
        
        # det prepare
        rois_norm = filters_detection.prepareInputs(rois, imageDims)
        
        # det predict
        start_pred = time.time()
        nb_det_rois = rois.shape[1]
        all_det_props = np.zeros((1,nb_det_rois, self.cfg.nb_object_classes))
        all_det_deltas = np.zeros((1,nb_det_rois, (self.cfg.nb_object_classes-1)*4))
        for batchidx in range(math.ceil(nb_det_rois / self.cfg.nb_detection_rois)):    
            sidx = batchidx * self.cfg.nb_detection_rois
            fidx = min(nb_det_rois, sidx + self.cfg.nb_detection_rois)
            batch_rois = np.zeros((1,self.cfg.nb_detection_rois, 5))
            batch_rois[0,:fidx-sidx,:] = rois_norm[:,sidx:fidx,:]
            det_props, det_deltas = self.model_det.predict_on_batch([self.shared_cnn, batch_rois])

            all_det_props[:,sidx:fidx,:] = det_props[:,:fidx-sidx,:]
            all_det_deltas[:,sidx:fidx,:] = det_deltas[:,:fidx-sidx,:]
        
        end_pred = time.time()
        
        # det post
        start_pp = time.time()
        rois = filters_detection.unprepareInputs(rois_norm, imageDims)

        if self.mode=='train' or self.model_hoi is not None or self.cfg.dataset=='TUPPMI':
            bboxes = helper.deltas2ObjBoxes(all_det_props, all_det_deltas, rois, imageDims, self.cfg, self.obj_mapping)
        else:
            bboxes = helper.deltas2Boxes(all_det_props, all_det_deltas, rois, imageDims, self.cfg)
        
        if len(bboxes) == 0:
            return None
        
        bboxes_nms = helper.non_max_suppression_boxes(bboxes, self.cfg, self.det_nms_thresh, max_boxes=10)
        bboxes_nms = np.expand_dims(bboxes_nms, axis=0)
        end_pp = time.time()
        
        if self.return_times:
            return bboxes_nms, [end_pred-start_pred, end_pp-start_pp]
        else:
            return bboxes_nms
        
    def stagethree_targets(self, X, imageMeta, imageDims):
        bboxes = X[0,::]
        all_hbboxes, all_obboxes, all_target_labels, val_map = filters_hoi.createTargets(bboxes, imageMeta, imageDims, self.cfg, self.obj_mapping)
        return all_hbboxes, all_obboxes, all_target_labels, val_map
    
    def stagethree(self, X, imageMeta, imageDims, obj_mapping=None, include='all'):
        # hoi prepare
        if len(X)==3:
            [self.shared_img, all_hbboxes, all_obboxes] = X
            self.shared_cnn = self.shared_img
        else:
            if len(X)==2:
                [self.shared_img, bboxes] = X
                self.shared_cnn = self.shared_img
            else:
                [bboxes] = X
            bboxes = np.copy(bboxes)
            all_hbboxes, all_obboxes = filters_hoi.splitInputs(bboxes, imageMeta, self.obj_mapping, self.hoi_mapping)
        
        if all_hbboxes is None:
            return None, None, None

        patterns = filters_hoi.createInteractionPatterns(all_hbboxes, all_obboxes, self.cfg)
        if not self.cfg.do_fast_hoi:
            hcrops, ocrops = filters_hoi.convertBB2Crop(self.shared_img, all_hbboxes, all_obboxes, imageDims)
        hbboxes_norm, obboxes_norm = filters_hoi.prepareInputs(all_hbboxes, all_obboxes, imageDims)
        
        # hoi predict
        start_pred = time.time()
        nb_hoi_rois = hbboxes_norm.shape[1]
#        nb_hoi_rois = self.cfg.nb_hoi_rois
        all_hoi_props = np.zeros((1,nb_hoi_rois, self.cfg.nb_hoi_classes))
        all_hoi_hbboxes = np.zeros((1,nb_hoi_rois, 5))
        all_hoi_obboxes = np.zeros((1,nb_hoi_rois, 5))
        for batchidx in range(math.ceil(nb_hoi_rois / self.cfg.nb_hoi_rois)):    
            sidx = batchidx * self.cfg.nb_hoi_rois
            fidx = min(nb_hoi_rois, sidx + self.cfg.nb_hoi_rois)
            batch_h = np.zeros((1,self.cfg.nb_hoi_rois,5))
            batch_o = np.zeros((1,self.cfg.nb_hoi_rois,5))
            batch_p = np.zeros((1,self.cfg.nb_hoi_rois, self.cfg.winShape[0], self.cfg.winShape[1], 2))
            batch_h[0,:fidx-sidx,:] = hbboxes_norm[:,sidx:fidx,::]
            batch_o[0,:fidx-sidx,:] = obboxes_norm[:,sidx:fidx,::]
            batch_p[0,:fidx-sidx,::] = patterns[:,sidx:fidx,:,:,:]
            
            if self.cfg.do_fast_hoi:  
                batch = [self.shared_cnn, batch_h[:,:fidx-sidx,::], batch_o[:,:fidx-sidx,::], batch_p[:,:fidx-sidx,::]]
            else:
                batch_hcrop = hcrops[sidx:fidx,::]
                batch_ocrop = ocrops[sidx:fidx,::]
                batch = [batch_hcrop, batch_ocrop, batch_p[0,:fidx-sidx,::], batch_h[0,:fidx-sidx,::], batch_o[0,:fidx-sidx,::]]
                        
            hoi_props, hoi_hbboxes, hoi_obboxes = self.model_hoi.predict_on_batch(batch)
            
            if not self.cfg.do_fast_hoi:
                hoi_props = np.expand_dims(hoi_props, axis=0)
                hoi_hbboxes = np.expand_dims(hoi_hbboxes, axis=0)
                hoi_obboxes = np.expand_dims(hoi_obboxes, axis=0)

            all_hoi_props[:,sidx:fidx,:] = hoi_props[:,:fidx-sidx,:]
            all_hoi_hbboxes[:,sidx:fidx,:] = hoi_hbboxes[:,:fidx-sidx,:]
            all_hoi_obboxes[:,sidx:fidx,:] = hoi_obboxes[:,:fidx-sidx,:]
        
        end_pred = time.time()
        
        # hoi post
        all_hoi_hbboxes, all_hoi_obboxes = filters_hoi.unprepareInputs(all_hoi_hbboxes, all_hoi_obboxes, imageDims)
        
        if self.return_times:
            return all_hoi_hbboxes[0], all_hoi_obboxes[0], all_hoi_props[0], [end_pred-start_pred]
        else:
            return all_hoi_hbboxes[0], all_hoi_obboxes[0], all_hoi_props[0]