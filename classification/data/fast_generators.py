# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:11:44 2018

@author: aag14
"""

import numpy as np
import random as r
import math as m
import cv2 as cv
import utils
import image
import copy as cp

class DataGenerator():
    
    def __init__(self, imagesMeta, GTMeta, labels, cfg, data_type='train'):
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
      self.images_per_batch = g_cfg.images_per_batch
      self.nb_samples_per_image = int(self.batch_size / self.images_per_batch)

      self.shuffle = g_cfg.shuffle
      self.inputs = cfg.inputs
      
      cfg.img_out_reduction = (16, 16)
      
      self.data_path = cfg.data_path
      self.images_path = self.data_path + 'images/'
      self.images_path = self.images_path + 'test/' if self.data_type == 'test' else self.images_path + 'train/'
      self.nb_classes = cfg.nb_classes
      self.cfg = cfg

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      else:
          self.dataID.sort()
      
      self.imagesMeta = imagesMeta
      self.GTMeta     = GTMeta
#      self.gt_label, _, _ = image.getYData(self.dataID, self.imagesMeta, self.GTMeta, self.cfg)
      self.nb_images = len(self.dataID)
#      self.nb_samples = len(self.gt_label)
      stats, _ = utils.getLabelStats(imagesMeta, labels)
      self.nb_samples = stats['totalx']
      if self.nb_batches is None:
          self.nb_batches = self.nb_images
      
    def getYData(self):
        gt_label, _, _ = image.getYData(self.dataID, self.imagesMeta, self.GTMeta, self.cfg, 32)
        gt_label = gt_label[0,:,:]
        return gt_label

        
      
    def begin(self):
        'Generates batches of samples'
        g = self._generateIterativeImageCentricBatches
        return g()
    
    def _generateBatchFromIDs(self, imageIdxs, batchIdx):
        imagesID = [self.dataID[idx] for idx in imageIdxs]
        
        imageMeta = self.imagesMeta[imageIdxs[0]]
        imageMeta = cp.copy(imageMeta)
        
        [dataXI, dataXH, dataXO], _ = image.getXData(imageMeta, self.images_path, self.cfg, batchIdx)
        dataXW = image.getDataPairWiseStream(imageMeta, self.cfg)            
        dataXW = np.expand_dims(dataXW, axis=0)
        y, _, _ = image.getYData(imagesID, self.imagesMeta, self.GTMeta, self.cfg)
        
        if dataXH.shape[1] > 32:
            dataXH = dataXH[:,:32,:]
            dataXO = dataXO[:,:32,:]
            dataXW = dataXW[:,:32,:,:,:]
            y      = y[:,:32,:]
        
        X = [dataXI, dataXH, dataXO, dataXW]        
        X = [X[i+1] for i in range(len(X)-1) if self.inputs[i]]
        if self.inputs[0] or self.inputs[1]:
            X = [dataXI] + X
            
        return X, y
		
    #%% Different forms of generators     
    def _generateIterativeImageCentricBatches(self):
        'Generates iterative batches of samples'
        
        while 1:
          imageIdx = 1
          # Generate batches
          for i in range(self.nb_batches):
              X = []; y = []
              batchIdx = 0
              for imageIdx in range(imageIdx-1, self.nb_images):
                  imageX, imageY = self._generateBatchFromIDs([imageIdx], batchIdx)  
                  X = utils.concatXData(X, imageX)
                  y.extend(imageY)
                  batchIdx += 1
                  if len(y) == self.images_per_batch or True:
                      break
                  
              y = np.array(y)
              yield X, y
    
