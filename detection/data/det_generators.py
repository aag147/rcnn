# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:11:44 2018

@author: aag14
"""

import numpy as np
import random as r
import math as m
import cv2 as cv
import filters_detection,\
       filters_rpn
import time
import utils
import os

class DataGenerator():
    
    def __init__(self, imagesMeta, cfg, data_type='train', do_meta=True, mode='train'):
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
      print('   inputs:', cfg.my_input_path + 'proposals_'+data_type)
      if os.path.exists(self.rois_path + 'proposals_'+data_type + '.pkl'):
          self.imagesInputs = utils.load_obj(self.rois_path + 'proposals_'+data_type)
          self.doIndyInputs = False
      else:
          self.doIndyInputs = True
      
      self.nb_images = len(self.dataID)
      self.nb_samples = None
      if self.nb_batches is None:
          self.nb_batches = self.nb_images
      
      
      
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
            imageInputs = utils.load_obj(self.rois_path + self.data_type + '/' + imageID)
        else:
            imageInputs = self.imagesInputs[imageID]
        return imageInputs
    
    def _generateBatchFromIDs(self, imageIdxs):
        imageIDs = [self.dataID[idx] for idx in imageIdxs]
#        if imageIDs[0] < '550394':
#            return None, None, None, None, None
#        else:
#            print(imageIDs)
        # Only works for batch_size=0
#        batchIdx = 0
        for imageID in imageIDs:
            imageMeta = self.imagesMeta[imageID]
            imageInputs = self._getImageInputs(imageID)
            imageMeta['id'] = imageID
            
            io_start = time.time()
            img, imageDims = filters_rpn.prepareInputs(imageMeta, self.images_path, self.cfg)
            io_end = time.time()
            
            pp_start = time.time()
#            Y_tmp = filters_detection.loadData(imageMeta, self.rois_path, imageDims, self.cfg)
            Y_tmp = filters_detection.loadData(imageInputs, self.cfg)
            if Y_tmp is None:
                raise Exception("ups: no detections available, path:%s" % self.rois_path)
            pp_end = time.time()
            
            if self.mode == 'test':
                bboxes, target_labels, target_deltas = Y_tmp
                return [img, bboxes], [target_labels, target_deltas], imageMeta, imageDims, None
            
#            bboxes, target_labels, target_deltas = Y_tmp
            bboxes, target_labels, target_deltas = filters_detection.reduceData(Y_tmp, self.cfg)
            bboxes = filters_detection.prepareInputs(bboxes, imageDims, imageMeta) 
            times = np.array([io_end-io_start, pp_end-pp_start])
            
        if self.do_meta:
            return [img, bboxes], [target_labels, target_deltas], imageMeta, imageDims, times
        return [img, bboxes], [target_labels, target_deltas]

    #%% Different forms of generators     
    def _generateIterativeImageCentricBatches(self):
        'Generates iterative batches of samples'
        imageIdxs = np.array(range(self.nb_images))
        while 1:
          r.shuffle(imageIdxs)
          for i in range(self.nb_images):
              imageIdx = imageIdxs[i]
              data = self._generateBatchFromIDs([imageIdx])
              if data is None:
                  continue
              yield data
              
    def _generateRandomImageCentricBatches(self):
        'Generates iterative batches of samples'
        
        while 1:
          imageIdxs = [r.randint(0, self.nb_images-1)]
          data = self._generateBatchFromIDs(imageIdxs)
          if data is None:
              continue
          yield data
    
