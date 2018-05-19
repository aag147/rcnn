# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:11:44 2018

@author: aag14
"""

import numpy as np
import random as r
import math as m
import cv2 as cv
import filters_rpn
import time

class DataGenerator():
    
    def __init__(self, imagesMeta, cfg, data_type='train'):
      'Initialization'
      self.data_type = data_type
      if data_type == 'train':
          g_cfg = cfg.train_cfg
      elif data_type == 'val':
          g_cfg = cfg.val_cfg
      else:
          g_cfg = cfg.test_cfg
          
      self.gen_type = g_cfg.type
      self.batch_size = g_cfg.batch_size
      self.nb_batches = g_cfg.nb_batches

      self.shuffle = g_cfg.shuffle
      self.inputs = cfg.inputs
      
      cfg.img_out_reduction = (16, 16)
      
      self.data_path = cfg.data_path
      self.images_path = self.data_path + 'images/'
      self.images_path = self.images_path + self.data_type + '/'
      self.cfg = cfg

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      else:
          self.dataID.sort()
      
      self.imagesMeta = imagesMeta
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
    
    def _generateBatchFromIDs(self, imageIdxs):
        imageIDs = [self.dataID[idx] for idx in imageIdxs]
        
        # Only works for batch_size=0
#        batchIdx = 0
        for imageID in imageIDs:
            imageMeta = self.imagesMeta[imageID]
            io_start = time.time()
            img, imageDims = filters_rpn.prepareInputs(imageMeta, self.images_path, self.cfg)
            io_end = time.time()
            pp_start = time.time()
            y_rpn_cls, y_rpn_regr = filters_rpn.prepareTargets(imageMeta, imageDims, self.cfg)
            pp_end = time.time()
            times = np.array([io_end-io_start, pp_end-pp_start])
            
        return img, [y_rpn_cls, y_rpn_regr], imageMeta, imageDims, times

    #%% Different forms of generators     
    def _generateIterativeImageCentricBatches(self):
        'Generates iterative batches of samples'
        
        while 1:
          currImageIdx = 0
          for i in range(self.nb_batches):
              imageIdxs = [imageIdx for imageIdx in range(currImageIdx, currImageIdx+self.batch_size)]
              currImageIdx += len(imageIdxs)
              data = self._generateBatchFromIDs(imageIdxs)
              yield data
              
    def _generateRandomImageCentricBatches(self):
        'Generates iterative batches of samples'
        
        while 1:
          imageIdxs = [r.randint(0, self.nb_images)]
          data = self._generateBatchFromIDs(imageIdxs)
          yield data
    
