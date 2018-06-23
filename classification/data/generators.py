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
import random
import copy as cp

class DataGenerator():
    
    def __init__(self, imagesMeta, GTMeta, cfg, data_type='train', do_meta=False):
      'Initialization'
      self.do_meta = do_meta
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
      
      cfg.img_out_reduction = (1, 1)
      
      self.data_path = cfg.data_path
      self.images_path = self.data_path + 'images/'
      self.images_path = self.images_path + 'test/' if self.data_type == 'test' else self.images_path + 'train/'
      self.nb_classes = cfg.nb_classes
      self.cfg = cfg

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      
      self.imagesMeta = imagesMeta
      self.GTMeta     = GTMeta
      gt_label, _, _ = image.getYData(self.dataID, self.imagesMeta, self.GTMeta, self.cfg)
      self.nb_images = len(self.dataID)
      self.nb_samples = gt_label.shape[1]
      if self.nb_batches is None:
          self.nb_batches = m.ceil(self.nb_samples / self.batch_size)
      
      
      
    def begin(self):
        'Generates batches of samples'
        if self.gen_type == 'rand':
            print('rand')
            g = self._generateRandomBatches
        else:
            g = self._generateIterativeBatches
        return g()
    
    def _generateBatchFromIDs(self, batchID):
        batchID = [self.dataID[idx] for idx in batchID]
#        print(batchID)
        imageMeta = self.imagesMeta[batchID[0]]
        imageMeta = cp.copy(imageMeta)
        
        [dataXP, dataXB], img = image.getX2Data(imageMeta, self.images_path, self.cfg)
        dataXW = image.getDataPairWiseStream(imageMeta, self.cfg)
        X = [dataXP, dataXB, dataXW]
        X = [X[i] for i in range(len(X)) if self.inputs[i]]
        y, _, _ = image.getYData(batchID, self.imagesMeta, self.GTMeta, self.cfg)
        y = y[0]
        
        if self.do_meta:
            return X, y, imageMeta, img
        
        return X, y
    
    def _generateBatchFromBGs(self, imageMeta, bbs):
        imageID = 'id'
        imagesMeta = {imageID: {'imageName': imageMeta['imageName']}}
        rels = {}
        for i in range(0, len(bbs), 2):
            rels = {str(i): {'objBB': bbs[i], 'prsBB': bbs[i+1], 'labels': [0]}}
        imagesMeta[imageID]['rels'] = rels
        
        [dataXP, dataXB] = image.getX2Data([imageID], imagesMeta, self.images_path, self.cfg)
        dataXW = image.getDataPairWiseStream([imageID], self.imagesMeta, self.cfg)
        X = [dataXP, dataXB, dataXW]
        y = np.zeros([int(len(bbs)/2), self.nb_classes])
        return X, y

    #%% Different forms of generators
    def _generateRandomBatches(self):
        'Generates random batches of samples'
        while 1:
          # Generate batches
              
          X = []
          y = np.zeros([self.batch_size, self.nb_classes])
          imageIdxs = r.sample(range(0,self.nb_images), self.images_per_batch)
          for idx, imageIdx in enumerate(imageIdxs):
              imageX, imageY = self._generateBatchFromIDs([imageIdx])
              # Variables
              s_idx = 0; f_idx = min(2,len(imageY))
              nb_bgs = self.nb_samples_per_image - f_idx
              idx = idx*self.nb_samples_per_image
              # Data
              imageMeta = self.imagesMeta[self.dataID[imageIdx]]
              imageXCut = utils.spliceXData(imageX, s_idx, f_idx)
              imageYCut = imageY[s_idx:f_idx, :]
              if nb_bgs > 0:
                  bbs  = utils.createBackgroundBBs(imageMeta, nb_bgs, self.images_path)
                  imageXBG, imageYBG  = self._generateBatchFromBGs(imageMeta, bbs)
                  imageXCmb = utils.concatXData(imageXCut, imageXBG)
#                  print(y.shape, idx, self.nb_samples_per_image, len(imageIdxs))
                  imageYCmb = np.append(imageYCut, imageYBG, 0)
                  
              else:
                  imageXCmb = imageXCut
                  imageYCmb = imageYCut
              X = utils.concatXData(X, imageXCmb)
              y[idx:idx+self.nb_samples_per_image, :] = imageYCmb
          yield X, y
          
    def _generateIterativeBatches(self):
        'Generates iterative batches of samples'
        
        
        while 1:
          imageIdx = 1
          hoiinimageidx = 0
          # Generate batches
          for i in range(self.nb_batches):
              X = []; y = []; imgs = []
              imageIdxs = []
              for imageIdx in range(imageIdx-1, self.nb_images):
                  imageIdxs.append(imageIdx)
                  if self.do_meta:
                      imageX, imageY, imageMeta, img = self._generateBatchFromIDs([imageIdx])    
                  else:
                      imageX, imageY = self._generateBatchFromIDs([imageIdx])
                      img = None
                  s_idx = 0; f_idx = len(imageY)
                  if hoiinimageidx > 0:
                      s_idx = hoiinimageidx
                      if hoiinimageidx == len(imageY):
                          hoiinimageidx = 0
                          continue
#                      hoiinimageidx = 0
                      
                  if (len(imageY) - hoiinimageidx) + len(y) >= self.batch_size:
                      hoiinimageidx = hoiinimageidx + len(imageY) - ((len(imageY) + len(y)) - self.batch_size)
                      f_idx = hoiinimageidx
                  else:
                     hoiinimageidx = 0
                      
                  imageXCut = utils.spliceXData(imageX, s_idx, f_idx)
                  X = utils.concatXData(X, imageXCut)
                  y.extend(imageY[s_idx:f_idx, :])
                  imgs.append(img)
#                  y.extend([self.dataID[imageIdx] for i in range(s_idx, f_idx)])
                  if len(y) == self.batch_size:
                      break
              imageIdx += 1
              y = np.array(y)
              
              if self.do_meta:
                  yield X, y, imageMeta, imgs[0]
              else:
                  yield X, y
    