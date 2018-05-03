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

class DataGenerator():
    
    def __init__(self, imagesMeta, GTMeta, cfg, data_type='train'):
      'Initialization'
      self.mean  = [103.939, 116.779, 123.68]
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
      self.gt_label, _, _ = image.getYData(self.dataID, self.imagesMeta, self.GTMeta, self.cfg)
      self.nb_images = len(self.dataID)
      self.nb_samples = None
      if self.nb_batches is None:
          self.nb_batches = self.nb_images
      
      
      
    def begin(self):
        'Generates batches of samples'
        if self.gen_type == 'rand':
            print('rand')
            g = self._generateRandomBatches
        else:
            g = self._generateIterativeImageCentricBatches
        return g()
    
    def _generateBatchFromIDs(self, imageIdx, batchIdx):
        batchID = [self.dataID[idx] for idx in imageIdx]
#        print(batchID)
        [dataXI, dataXH, dataXO], _ = image.getXData(batchID, self.imagesMeta, self.images_path, self.cfg, batchIdx)
        dataXW = image.getDataPairWiseStream(batchID, self.imagesMeta, self.cfg)
        X = [dataXI, dataXH, dataXO, dataXW]
        X = [X[i+1] for i in range(len(X)-1) if self.inputs[i]]
        if self.inputs[0] or self.inputs[1]:
            X = [dataXI] + X
        y, _, _ = image.getYData(batchID, self.imagesMeta, self.GTMeta, self.cfg)
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

     
    def _generateIterativeBatches(self):
        'Generates iterative batches of samples'
        
#        thisID = 'HICO_train2015_00016257.jpg'
        while 1:
          imageIdx = 1
          hoiinimageidx = 0
          # Generate batches
          for i in range(self.nb_batches):
              X = []; y = []
              imageIdxs = []
              batchIdx = 0
              for imageIdx in range(imageIdx-1, self.nb_images):
                  imageIdxs.append(imageIdx)
                  imageX, imageY = self._generateBatchFromIDs([imageIdx], batchIdx)
#                  imageY = np.array([self.dataID[imageIdx] for i in range(len(imageY))])
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
#                      hoiinimageidx += tmp_hoiinimageidx
                  else:
                     hoiinimageidx = 0
#                  if s_idx > 0 or f_idx != len(imageY):
#                      print('ID', self.dataID[imageIdx][18:], str(s_idx) + '/' + str(f_idx) + '/' + str(len(imageY)))
#                  if imageIdx > 1500:
#                      print(self.dataID[imageIdx])
                  imageXCut = utils.spliceXData(imageX, s_idx, f_idx)
                  X = utils.concatXData(X, imageXCut)
                  y.extend(imageY[s_idx:f_idx,:])
                  batchIdx += 1
                  if len(y) == self.batch_size:
                      break
#              if imageIdx > 1500:
#                  print('lengh', len(y))
              imageIdx += 1
              y = np.array(y)
#              print(X[0].shape, X[1].shape, X[2].shape, X[3].shape)
#              sys.stdout.write('\r' + str(X[0].shape) + str(X[1].shape) + ' - nbs: ' + str(i) + '-' + str(self.nb_batches))
#              sys.stdout.flush()
              
#              if self.inputs[0] or self.inputs[1]:
#                  X[1] = np.insert(X[1], 0, [idx for idx in range(self.batch_size)], axis=1)
#              if self.inputs[0] and self.inputs[1]:
#                  X[2] = np.insert(X[2], 0, [idx for idx in range(self.batch_size)], axis=1)
#              print('fin',min(X[1][:,0]), max(X[1][:,0]))
#              print(X[0].shape, y.shape)
              yield X, y
    
