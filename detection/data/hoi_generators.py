# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:11:44 2018

@author: aag14
"""

import numpy as np
import random as r
import math as m
import cv2 as cv
import filters_hoi,\
       filters_rpn
import time
import utils
import os
import sys

class DataGenerator():
    
    def __init__(self, imagesMeta, cfg, data_type='train', do_meta=True, mode='train', approach='new'):
      'Initialization'
      self.data_type = data_type
      if data_type == 'train':
          g_cfg = cfg.train_cfg
      elif data_type == 'val':
          g_cfg = cfg.val_cfg
      else:
          g_cfg = cfg.test_cfg
          
      self.mode = mode
      self.gen_type = g_cfg.type
      self.batch_size = g_cfg.batch_size
      self.nb_batches = g_cfg.nb_batches

      self.shuffle = g_cfg.shuffle
      self.inputs = cfg.inputs
      self.do_meta = do_meta
      self.approach = approach
      
      cfg.img_out_reduction = (16, 16)
      
      self.data_path = cfg.data_path
      self.images_path = self.data_path + 'images/'
      self.images_path = self.images_path + self.data_type + '/'
      self.rois_path = cfg.my_input_path
      self.cfg = cfg

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      else:
          self.dataID.sort()
      
      self.imagesMeta = imagesMeta
      if os.path.exists(self.rois_path + 'hoiputs_'+data_type + '.pkl'):
          self.imagesInputs = utils.load_obj(self.rois_path + 'hoiputs_'+data_type)
          self.doIndyInputs = False
      else:
          adir = 'newest' if self.approach == 'newest' else 'new'
          self.rois_path = self.rois_path + self.data_type + adir+ '/'
          assert os.path.exists(self.rois_path), self.rois_path
          self.doIndyInputs = True
      
      
      self.nb_images = len(self.dataID)
      self.nb_samples = None
      if self.nb_batches is None:
          self.nb_batches = self.nb_images
      
      
    def countValidImages(self):
        nb_batches = 0
        for imageID, imageInputs in self.imagesInputs.items():
            if imageInputs is None:
                continue

#            val_map = np.array(imageInputs['val_map'])
#            if len(np.where(val_map==3)[0])==0:
#                continue
            nb_batches += 1
            
        return nb_batches
    
      
    def begin(self):
        'Generates batches of samples'
        if self.gen_type == 'rand':
            print('Random images')
            g = self._generateRandomImageCentricBatches
        else:
            print('Iterate images')
            g = self._generateIterativeImageCentricBatches
        return g()
    
    
    def _getImageInputs(self, imageID):
        if self.doIndyInputs:
            assert os.path.exists(self.rois_path+imageID+'.pkl'), self.rois_path + imageID
            imageInputs = utils.load_obj(self.rois_path + imageID)
        else:
            imageInputs = self.imagesInputs[imageID]
        return imageInputs
    
    
    def _generateSlowBatch(self, imageIDs):
        for imageID in imageIDs:
            imageMeta = self.imagesMeta[imageID]            
            imageInputs = self._getImageInputs(imageID)
            
            imageMeta['id'] = imageID
            io_start = time.time()
            img, imageDims = filters_rpn.prepareInputs(imageMeta, self.images_path, self.cfg)
            io_end = time.time()
            pp_start = time.time()
            Y_tmp = filters_hoi.loadData(imageInputs, imageDims, self.cfg)
            pp_end = time.time()
            if Y_tmp is None:
                if self.mode == 'train':
                    raise Exception("ups: no detections available, path:%s" % self.rois_path)
                else:
                    return None
            
            if self.mode == 'val':
                all_hbboxes, all_obboxes, all_target_labels, all_val_map = Y_tmp
                if all_val_map.shape[1] > self.cfg.nb_hoi_rois:
                    idxs = np.random.choice(list(range(self.cfg.nb_hoi_rois)), self.cfg.nb_hoi_rois, replace=False)
                    all_hbboxes = all_hbboxes[:,idxs,:]
                    all_obboxes = all_obboxes[:,idxs,:]
                    all_target_labels = all_target_labels[:,idxs,:]
                patterns = filters_hoi.createInteractionPatterns(all_hbboxes, all_obboxes, self.cfg)
                hcrops, ocrops = filters_hoi.convertBB2Crop(img, all_hbboxes, all_obboxes, imageDims)
                all_hbboxes, all_obboxes = filters_hoi.prepareInputs(all_hbboxes, all_obboxes, imageDims)
                
                b_hcrops = np.zeros((self.cfg.nb_hoi_rois,)+hcrops[0].shape)
                b_hcrops[:hcrops.shape[0],::] = hcrops
                b_ocrops = np.zeros((self.cfg.nb_hoi_rois,)+ocrops[0].shape)
                b_ocrops[:hcrops.shape[0],::] = ocrops
                b_pattern = np.zeros((self.cfg.nb_hoi_rois,)+patterns[0][0].shape)
                b_pattern[:patterns.shape[1],::] = patterns
                b_hbboxes = np.zeros((self.cfg.nb_hoi_rois,)+all_hbboxes[0][0].shape)
                b_hbboxes[:all_hbboxes.shape[1],::] = all_hbboxes
                b_obboxes = np.zeros((self.cfg.nb_hoi_rois,)+all_obboxes[0][0].shape)
                b_obboxes[:all_obboxes.shape[1],::] = all_obboxes
                b_targets = np.zeros((self.cfg.nb_hoi_rois,)+all_target_labels[0][0].shape)
                b_targets[:all_target_labels.shape[1],::] = all_target_labels
                
                if self.do_meta:
                    return [hcrops, ocrops, patterns[0], all_hbboxes[0], all_obboxes[0]], all_target_labels[0], imageMeta, imageDims, None
                return [b_hcrops, b_ocrops, b_pattern, b_hbboxes, b_obboxes], b_targets       
            elif self.mode == 'test':
                all_hbboxes, all_obboxes, all_target_labels, all_val_map = Y_tmp
                return [img, all_hbboxes, all_obboxes, all_val_map], all_target_labels, imageMeta, imageDims, None
            
            hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, self.cfg)
            if hbboxes is None:
                return None

            patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, self.cfg)
            hcrops, ocrops = filters_hoi.convertBB2Crop(img, hbboxes, obboxes, imageDims)
            hbboxes, obboxes = filters_hoi.prepareInputs(hbboxes, obboxes, imageDims)
            
            times = np.array([io_end-io_start, pp_end-pp_start])
            
        if self.do_meta:
            return [hcrops, ocrops, patterns[0], hbboxes[0], obboxes[0]], target_labels[0], imageMeta, imageDims, times
        return [hcrops, ocrops, patterns[0], hbboxes[0], obboxes[0]], target_labels[0]       
    
    def _generateFastBatch(self, imageIDs):
        for imageID in imageIDs:
            imageMeta = self.imagesMeta[imageID]
            imageInputs = self._getImageInputs(imageID)
            
            imageMeta['id'] = imageID
            io_start = time.time()
            img, imageDims = filters_rpn.prepareInputs(imageMeta, self.images_path, self.cfg)
            io_end = time.time()
            pp_start = time.time()
            Y_tmp = filters_hoi.loadData(imageInputs, imageDims, self.cfg)
            pp_end = time.time()
            if Y_tmp is None:
                if self.mode == 'train':
                    raise Exception("ups: no detections available, path:%s" % self.rois_path)
                else:
                    return None
                
            if self.mode == 'val':
                all_hbboxes, all_obboxes, all_target_labels, all_val_map = Y_tmp
                if all_val_map.shape[1] > self.cfg.nb_hoi_rois:
                    idxs = np.random.choice(list(range(self.cfg.nb_hoi_rois)), self.cfg.nb_hoi_rois, replace=False)
                    all_hbboxes = all_hbboxes[:,idxs,:]
                    all_obboxes = all_obboxes[:,idxs,:]
                    all_target_labels = all_target_labels[:,idxs,:]
                patterns = filters_hoi.createInteractionPatterns(all_hbboxes, all_obboxes, self.cfg)
                all_hbboxes, all_obboxes = filters_hoi.prepareInputs(all_hbboxes, all_obboxes, imageDims)
                
                b_pattern = np.zeros((1,self.cfg.nb_hoi_rois)+patterns[0][0].shape)
                b_pattern[0,:patterns.shape[1],::] = patterns
                b_hbboxes = np.zeros((1,self.cfg.nb_hoi_rois)+all_hbboxes[0][0].shape)
                b_hbboxes[0,:all_hbboxes.shape[1],::] = all_hbboxes
                b_obboxes = np.zeros((1,self.cfg.nb_hoi_rois)+all_obboxes[0][0].shape)
                b_obboxes[0,:all_obboxes.shape[1],::] = all_obboxes
                b_targets = np.zeros((1,self.cfg.nb_hoi_rois)+all_target_labels[0][0].shape)
                b_targets[0,:all_target_labels.shape[1],::] = all_target_labels
                
                if self.do_meta:
                    return [img, all_hbboxes, all_obboxes, patterns], all_target_labels, imageMeta, imageDims, None
                return [img, b_hbboxes, b_obboxes, b_pattern], b_targets
                
            elif self.mode == 'test':
                all_hbboxes, all_obboxes, all_target_labels, all_val_map = Y_tmp
                return [img, all_hbboxes, all_obboxes, all_val_map], all_target_labels, imageMeta, imageDims, None
            
            hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, self.cfg)
            patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, self.cfg)
            hbboxes, obboxes = filters_hoi.prepareInputs(hbboxes, obboxes, imageDims)
            
            times = np.array([io_end-io_start, pp_end-pp_start])
            
        if self.do_meta:
            return [img, hbboxes, obboxes, patterns], target_labels, imageMeta, imageDims, times
        return [img, hbboxes, obboxes, patterns], target_labels
        
        
    def _generateBatchFromIDs(self, imageIdxs, list_idx):
        imageIDs = [self.dataID[idx] for idx in imageIdxs]
#        imageIDs = ['HICO_test2015_00008335']
#        utils.update_progress_new(list_idx+1, self.nb_batches, imageIDs[0])
        
        if self.cfg.do_fast_hoi:
            return self._generateFastBatch(imageIDs)
        else:
            return self._generateSlowBatch(imageIDs)
        
        



    #%% Different forms of generators     
    def _generateIterativeImageCentricBatches(self):
        'Generates iterative batches of samples'
        imageIdxs = np.array(range(self.nb_images))
        while 1:
          r.shuffle(imageIdxs)
          for i in range(self.nb_batches):
              imageIdx = imageIdxs[i]
              data = self._generateBatchFromIDs([imageIdx], i)
              if data is None:
                  if self.mode=='test':
                      data = [None, None, None, None], None, None, None, None
                  else:
                      continue
              yield data
              
    def _generateRandomImageCentricBatches(self):
        'Generates iterative batches of samples'
        
        while 1:
          imageIdxs = [r.randint(0, self.nb_images-1)]
          data = self._generateBatchFromIDs(imageIdxs)
#          if data is None:
#              continue
          yield data
    
