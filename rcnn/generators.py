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

class DataGenerator():
    
    def __init__(self, imagesMeta, cfg, data_type='train'):
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
      self.shuffle = g_cfg.shuffle
      #self.xdim = cfg.xdim
      #self.ydim = cfg.ydim
      #self.cdim = cfg.cdim
      self.shape = (cfg.xdim, cfg.ydim)
      self.cIdx = 0
      self.winShape = (64, 64)
      
      self.data_path = cfg.data_path
      self.nb_classes = cfg.nb_classes

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      
      self.imagesMeta = imagesMeta
      self.gt_label, self.gt_bb = utils.getYData(self.dataID, self.imagesMeta, self.nb_classes)
       
      self._configGenerators()
      
      
    def begin(self):
        'Generates batches of samples'
        if self.gen_type =='class-rand':
            g = self._generateLabelRandomBatches
        elif self.gen_type == 'class-itr':
            g = self._generateLabelIterativeBatches
        elif self.gen_type == 'rand':
            g = self._generateRandomBatches
        else:
            g = self._generateIterativeBatches
        return g()
    
    def _generateBatchFromIDs(self, batchID):
        batchID = [self.dataID[idx] for idx in batchID]
        data_path = self.data_path + '_images/'
        data_path = data_path + 'test/' if self.data_type == 'test' else data_path + 'train/'
        [dataXP, dataXB] = utils.getX2Data(batchID, self.imagesMeta, data_path, self.shape)
        dataXW = self.getDataPairWiseStream(batchID, self.imagesMeta)
        X = [dataXP, dataXB, dataXW]
        y, _ = utils.getYData(batchID, self.imagesMeta, self.nb_classes)
        return X, y

    def _generateRandomBatches(self):
        'Generates random batches of samples'
        nb_samples = len(self.dataID)
        while 1:
          # Generate batches
          batchID = r.sample(range(0,nb_samples), self.batch_size)     
          X, y = self._generateBatchFromIDs(batchID)
          yield X, y
    def _generateIterativeBatches(self):
        'Generates iterative batches of samples'
        nb_samples = len(self.dataID)
        hoiinimageidx = 0
        while 1:
          imageIdx = 1
          # Generate batches
          for i in range(self.nb_batches):
              X = []; y = []
              for imageIdx in range(imageIdx-1, min(nb_samples, imageIdx + self.batch_size)):
                  imageX, imageY = self._generateBatchFromIDs([imageIdx])
#                  print('x', imageX[0].shape)
#                  print('y', imageY.shape)
                  s_idx = 0; f_idx = len(imageY)
                  if hoiinimageidx > 0:
                      s_idx = hoiinimageidx
                      if hoiinimageidx == len(imageY):
                          hoiinimageidx = 0
                          continue
                      hoiinimageidx = 0
                      
                  if len(imageY) + len(y) >= self.batch_size:
                      hoiinimageidx = len(imageY) - ((len(imageY) + len(y)) - self.batch_size)
                      f_idx = hoiinimageidx
                      
                  imageXCut = utils.spliceXData(imageX, s_idx, f_idx)
#                  print('c', imageXCut[0].shape)
                  X = utils.concatXData(X, imageXCut)
                  y.extend(imageY[s_idx:f_idx, :])
#                  print('X', s_idx, f_idx, X[0].shape, imageIdx)
                  if len(y) == self.batch_size:
                      break
              imageIdx += 1
              y = np.array(y)
#              print('X', s_idx, f_idx, X[0].shape, y.shape, imageIdx)
              yield X, y
#              print(batchID)
    def _generateLabelIterativeBatches(self):
        'Generates label iterative batches of samples'
        # Infinite loop
        while 1:
        
          # Generate batches
          for i in range(self.nb_batches):
              batchID = []
              for y in range(self.nb_classes):
                  cond = self.gt_label[:,y]==1
                  indicies = np.argwhere(cond)
                  indicies = np.squeeze(indicies)
                  for j in range(self.samples_per_label_in_batch):
                      index = (self.samples_per_label_in_batch*i + j) % len(indicies)
                      batchID.append(indicies[index])

              X, y = self._generateBatchFromIDs(batchID)
              yield X, y


    def _configGenerators(self):
        if self.gen_type.startswith( 'class' ):
            self.samples_per_label_in_batch = 2
            self.batch_size = self.nb_classes*self.samples_per_label_in_batch
            C = np.sum(self.gt_label, axis=0)
            C[::-1].sort()
            maxC = C[self.cIdx]
            self.nb_batches = m.ceil(maxC/self.samples_per_label_in_batch)
        else:
            self.batch_size = self.batch_size
            self.nb_batches = m.ceil(len(self.gt_label) / self.batch_size)
            self.nb_samples = len(self.gt_label)
            
            
    def getSinglePairWiseStream(self, thisBB, thatBB, width, height, newWidth, newHeight):
        xmin = max(0, thisBB['xmin'] - thatBB['xmin'])
        xmax = width - max(0, thatBB['xmax'] - thisBB['xmax'])
        ymin = max(0, thisBB['ymin'] - thatBB['ymin'])
        ymax = height - max(0, thatBB['ymax'] - thisBB['ymax'])
        
        attWin = np.zeros([height,width])
        attWin[ymin:ymax, xmin:xmax] = 1
        attWin = cv.resize(attWin, (newWidth, newHeight), interpolation = cv.INTER_NEAREST)
        attWin = attWin.astype(np.int)

        xPad = int(abs(newWidth - self.winShape[0]) / 2)
        yPad = int(abs(newHeight - self.winShape[0]) / 2)
        attWinPad = np.zeros(self.winShape).astype(np.int)
#        print(attWin.shape, attWinPad.shape, xPad, yPad)
#        print(height, width, newHeight, newWidth)
        attWinPad[yPad:yPad+newHeight, xPad:xPad+newWidth] = attWin
        return attWinPad

    def getPairWiseStream(self, prsBB, objBB):
        width = max(prsBB['xmax'], objBB['xmax']) - min(prsBB['xmin'], objBB['xmin'])
        height = max(prsBB['ymax'], objBB['ymax']) - min(prsBB['ymin'], objBB['ymin'])
        if width > height:
            newWidth = self.winShape[0]
            apr = newWidth / width
            newHeight = int(height*apr) 
        else:
            newHeight = self.winShape[0]
            apr = newHeight / height
            newWidth = int(width*apr)
            
        prsWin = self.getSinglePairWiseStream(prsBB, objBB, width, height, newWidth, newHeight)
        objWin = self.getSinglePairWiseStream(objBB, prsBB, width, height, newWidth, newHeight)
        
        return [prsWin, objWin]

    def getDataPairWiseStream(self, imagesID, imagesMeta):
        dataPar = []
        for imageID in imagesID:
            imageMeta = imagesMeta[imageID]
            for relID, rel in imageMeta['rels'].items():
                relWin = self.getPairWiseStream(rel['prsBB'], rel['objBB'])
                dataPar.append(relWin)
        dataPar = np.array(dataPar)
        return dataPar
    