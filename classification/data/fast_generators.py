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
import sys
import filters_helper as helper


class DataGenerator():
    
    def __init__(self, imagesMeta, GTMeta, labels, cfg, data_type='train'):
      'Initialization'
      self.mean  = cfg.img_channel_mean
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
      stats, _ = utils.getLabelStats(GTMeta, labels)
      self.nb_samples = stats['total']
      if self.nb_batches is None:
          self.nb_batches = self.nb_images
      
      
        
    def getXData(self, imagesID, batchIdx):
        dataX = []
        dataH = []
        dataO = []
        IDs   = []
        for imageID in imagesID:
            imageMeta = self.imagesMeta[imageID]
            img = cv.imread(self.images_path + imageMeta['imageName'])
    #        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            assert(img is not None)
            assert(img.shape[0] > 10)
            assert(img.shape[1] > 10)
            assert(img.shape[2] == 3)
            imgRedux, scale = helper.preprocessImage(img, self.cfg)
    
            tmpH = []
            tmpO = []
            for relID, rel in imageMeta['rels'].items():
                h, o = image.getDataFromRel(rel['prsBB'], rel['objBB'], scale, [0,0], imgRedux.shape, self.cfg)
                tmpH.append([batchIdx] + h)
                tmpO.append([batchIdx] + o)
                
            dataX.append(imgRedux)
            dataH.append(tmpH)
            dataO.append(tmpO)
            IDs.append(imageID)
            batchIdx += 1
        dataX = np.array(dataX)
        dataH = np.array(dataH)
        dataO = np.array(dataO)
    #    print(dataX.shape, dataH.shape, dataO.shape)
        IDs = np.array(IDs)
        
    #    dataH = np.expand_dims(dataH, axis=1)
    #    dataO = np.expand_dims(dataO, axis=1)
        
        return [dataX, dataH, dataO], IDs
        
      
    def begin(self):
        'Generates batches of samples'
        g = self._generateIterativeImageCentricBatches
        return g()
    
    def _generateBatchFromIDs(self, imageIdxs, batchIdx):
        imagesID = [self.dataID[idx] for idx in imageIdxs]
        [dataXI, dataXH, dataXO], _ = self.getXData(imagesID, batchIdx)
        dataXW = image.getDataPairWiseStream(imagesID, self.imagesMeta, self.cfg)            
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
    
