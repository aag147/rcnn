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

class DataGenerator():
    
    def __init__(self, imagesMeta, cfg, data_type='train', do_meta=True):
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
      self.do_meta = do_meta
      
      cfg.img_out_reduction = (16, 16)
      
      self.data_path = cfg.data_path
      self.images_path = self.data_path + 'images/'
      self.images_path = self.images_path + self.data_type + '/'
      self.rois_path = cfg.my_detections_path
      self.cfg = cfg

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      else:
          self.dataID.sort()
      
      self.imagesMeta = imagesMeta
      self.imagesInputs = utils.load_dict(self.rois_path + 'hoiputs')
      
      
      self.nb_images = len(self.dataID)
      self.nb_samples = None
      if self.nb_batches is None:
          self.nb_batches = self.countValidImages()
      
      
    def countValidImages(self):
        nb_batches = 0
        for imageID, imageInputs in self.imagesInputs.items():
            if imageInputs is None:
                continue

            val_map = np.array(imageInputs['val_map'])
            if len(np.where(val_map==3)[0])==0:
                continue
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
    
    def _generateBatchFromIDs(self, imageIdxs):
        imageIDs = [self.dataID[idx] for idx in imageIdxs]

        for imageID in imageIDs:
            imageMeta = self.imagesMeta[imageID]
            imageInputs = self.imagesInputs[imageID]
            
            if imageInputs is None:
                return None
            
            imageMeta['id'] = imageID
            io_start = time.time()
            img, imageDims = filters_rpn.prepareInputs(imageMeta, self.images_path, self.cfg)
            io_end = time.time()
            pp_start = time.time()
            Y_tmp = filters_hoi.loadData(imageInputs, imageDims, self.cfg)
            pp_end = time.time()
            if Y_tmp[0] is None:
                return None
            hbboxes, obboxes, target_labels = filters_hoi.reduceTargets(Y_tmp, self.cfg)
            patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, self.cfg)
            hbboxes, obboxes = filters_hoi.prepareInputs(hbboxes, obboxes, imageDims)
            
            times = np.array([io_end-io_start, pp_end-pp_start])
            
        if self.do_meta:
            return [img, hbboxes, obboxes, patterns], target_labels, imageMeta, imageDims, times
        return [img, hbboxes, obboxes, patterns], target_labels

    #%% Different forms of generators     
    def _generateIterativeImageCentricBatches(self):
        'Generates iterative batches of samples'
        imageIdxs = np.array(range(self.nb_images))
        while 1:
          r.shuffle(imageIdxs)
          for i in range(self.nb_batches):
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
#          if data is None:
#              continue
          yield data
    