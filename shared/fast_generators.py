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
      
      cfg.shape = (cfg.ydim, cfg.xdim)
      cfg.order_of_dims = [0,1,2]
      cfg.par_order_of_dims = [0,2,3,1]
      cfg.winShape = (64, 64)
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
      self.gt_label, self.gt_bb = image.getYData(self.dataID, self.imagesMeta, self.GTMeta, self.cfg)
      self.nb_images = len(self.dataID)
      self.nb_samples = len(self.gt_label)
      if self.nb_batches is None:
          self.nb_batches = m.floor(self.nb_samples / self.batch_size)
      
      
      
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
        [dataXI, dataXH, dataXO], _ = image.getXData(batchID, self.imagesMeta, self.images_path, self.cfg)
        dataXW = image.getDataPairWiseStream(batchID, self.imagesMeta, self.cfg)
        X = [dataXI, dataXH, dataXO, dataXW]
#        X = [X[i] for i in range(len(X)) if self.inputs[i]]
        y, _ = image.getYData(batchID, self.imagesMeta, self.GTMeta, self.cfg)
        return X, y

    #%% Different forms of generators          
    def _generateIterativeBatches(self):
        'Generates iterative batches of samples'
        hoiinimageidx = 0
        
#        thisID = 'HICO_train2015_00016257.jpg'
        while 1:
          imageIdx = 1
          # Generate batches
          for i in range(self.nb_batches):
              X = []; y = []
              imageIdxs = []
              for imageIdx in range(imageIdx-1, self.nb_images):
                  imageIdxs.append(imageIdx)
                  imageX, imageY = self._generateBatchFromIDs([imageIdx])
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
                  if len(y) == self.batch_size:
                      break
#              if imageIdx > 1500:
#                  print('lengh', len(y))
              imageIdx += 1
              y = np.array(y)
#              print(X[0].shape, X[1].shape, X[2].shape, X[3].shape)
#              sys.stdout.write('\r' + str(X[0].shape) + str(X[1].shape) + ' - nbs: ' + str(i) + '-' + str(self.nb_batches))
#              sys.stdout.flush()
              yield X, y
    
