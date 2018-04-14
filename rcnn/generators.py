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
      self.nb_batches = g_cfg.nb_batches
      self.images_per_batch = g_cfg.images_per_batch
      self.nb_samples_per_image = int(self.batch_size / self.images_per_batch)

      self.shuffle = g_cfg.shuffle
      self.xdim = cfg.xdim
      self.ydim = cfg.ydim
      self.cdim = cfg.cdim
      self.shape = (cfg.ydim, cfg.xdim)
      self.cIdx = 0
      self.winShape = (64, 64)
      
      self.data_path = cfg.data_path
      self.images_path = self.data_path + 'images/'
      self.images_path = self.images_path + 'test/' if self.data_type == 'test' else self.images_path + 'train/'
      self.nb_classes = cfg.nb_classes

      self.dataID = list(imagesMeta.keys())
      self.dataID.sort()
      if self.shuffle:
          r.shuffle(self.dataID)
      
      self.imagesMeta = imagesMeta
      self.gt_label, self.gt_bb = utils.getYData(self.dataID, self.imagesMeta, self.nb_classes)
      self.nb_images = len(self.dataID)
      self.nb_samples = len(self.gt_label)
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
        [dataXP, dataXB] = utils.getX2Data(batchID, self.imagesMeta, self.images_path, self.shape)
        dataXW = self.getDataPairWiseStream(batchID, self.imagesMeta)
        X = [dataXP, dataXB, dataXW]
        y, _ = utils.getYData(batchID, self.imagesMeta, self.nb_classes)
        return X, y
    
    def _generateBatchFromBGs(self, imageMeta, bbs):
        imageID = 'id'
        imagesMeta = {imageID: {'imageName': imageMeta['imageName']}}
        rels = {}
        for i in range(0, len(bbs), 2):
            rels = {str(i): {'objBB': bbs[i], 'prsBB': bbs[i+1], 'labels': [0]}}
        imagesMeta[imageID]['rels'] = rels
        
        [dataXP, dataXB] = utils.getX2Data([imageID], imagesMeta, self.images_path, self.shape)
        dataXW = self.getDataPairWiseStream([imageID], imagesMeta)
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
        hoiinimageidx = 0
        while 1:
          imageIdx = 1
          # Generate batches
          for i in range(self.nb_batches):
              X = []; y = []
              imageIdxs = []
              for imageIdx in range(imageIdx-1, self.nb_images):
                  imageIdxs.append(imageIdx)
                  imageX, imageY = self._generateBatchFromIDs([imageIdx])
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
                  X = utils.concatXData(X, imageXCut)
                  y.extend(imageY[s_idx:f_idx, :])
                  if len(y) == self.batch_size:
                      break
              imageIdx += 1
              y = np.array(y)
              yield X, y
            
    #%% Special third stream data extraction
    def _getSinglePairWiseStream(self, thisBB, thatBB, width, height, newWidth, newHeight):
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

    def _getPairWiseStream(self, prsBB, objBB):
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
            
        prsWin = self._getSinglePairWiseStream(prsBB, objBB, width, height, newWidth, newHeight)
        objWin = self._getSinglePairWiseStream(objBB, prsBB, width, height, newWidth, newHeight)
        
        return [prsWin, objWin]

    def getDataPairWiseStream(self, imagesID, imagesMeta):
        dataPar = []
        for imageID in imagesID:
            imageMeta = imagesMeta[imageID]
            for relID, rel in imageMeta['rels'].items():
                relWin = self._getPairWiseStream(rel['prsBB'], rel['objBB'])
                dataPar.append(relWin)
        dataPar = np.array(dataPar)
        return dataPar
    