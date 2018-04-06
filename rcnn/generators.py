# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:11:44 2018

@author: aag14
"""

import numpy as np
import random as r
import preprocess as pp
from HO_RCNN import HO_RCNN as ho
import cv2 as cv

class DataGenerator():
    
    def __init__(self, train_type='itr', val_type='itr', batch_size=32, xdim=227, ydim=227, cdim=3, shuffle=False):
      'Initialization'
      self.mean  = [103.939, 116.779, 123.68]
      self.train_type = train_type
      self.val_type   = val_type
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.xdim = xdim
      self.ydim = ydim
      self.cdim = cdim
      self.cIdx = 0
      self.winShape = (64, 64)
      
    def setDataVariables(self, allID, trainID, valID, testID, imagesMeta, images, unique_labels):
        self.imagesMeta = imagesMeta
        self.images = images
        self.unique_labels = unique_labels
        self.nb_classes = len(unique_labels)
        
        self.allID = allID
        self.trainID = trainID
        self.valID = valID
        self.testID = testID
        
        self._prepareData()
        self._configGenerators()
        
    def _normalizeImages(self):        
        self.meanPrs = self.trainX[0].mean()
        self.meanObj = self.trainX[1].mean()
        
        self.stdPrs = self.trainX[0].std()
        self.stdObj = self.trainX[1].std()
        
        self.trainX[0] -= self.meanPrs
        self.trainX[0] /= self.stdPrs
        self.trainX[1] -= self.meanObj
        self.trainX[1] /= self.stdObj
        
        self.valX[0] -= self.meanPrs
        self.valX[0] /= self.stdPrs
        self.valX[1] -= self.meanObj
        self.valX[1] /= self.stdObj
        
        self.testX[0] -= self.meanPrs
        self.testX[0] /= self.stdPrs
        self.testX[1] -= self.meanObj
        self.testX[1] /= self.stdObj
        
    def _prepareData(self):                
        [trainPrsX, trainObjX, trainY, _] = pp.getData(self.trainID, self.imagesMeta, self.images, (self.xdim, self.ydim))
        [valPrsX, valObjX, valY, _]       = pp.getData(self.valID, self.imagesMeta, self.images, (self.xdim, self.ydim))
        [testPrsX, testObjX, testY, _]    = pp.getData(self.testID, self.imagesMeta, self.images, (self.xdim, self.ydim))
        
        self.trainY = trainY
        self.valY = valY
        self.testY = testY        
        
        trainParX = self.getDataPairWiseStream(self.trainID, self.imagesMeta)
        valParX = self.getDataPairWiseStream(self.valID, self.imagesMeta)
        testParX = self.getDataPairWiseStream(self.testID, self.imagesMeta)
        
        self.trainYMatrix = pp.getMatrixLabels(self.unique_labels, trainY)
        self.valYMatrix = pp.getMatrixLabels(self.unique_labels, valY)
        self.testYMatrix = pp.getMatrixLabels(self.unique_labels, testY)
        
        self.trainX = [trainPrsX, trainObjX, trainParX]
        self.valX   = [valPrsX, valObjX, valParX]
        self.testX  = [testPrsX, testObjX, testParX]
        
        self._normalizeImages()
      
      
    def generateTrain(self):
        return self._generate(self.trainX, self.trainY, self.nb_train_batches, self.train_type)
    def generateVal(self, gen_type='itr'):
        return self._generate(self.valX, self.valY, self.nb_val_batches, self.val_type)
      
    def _generate(self, dataX, dataY, nb_batches, gen_type):
        'Generates batches of samples'
        if gen_type =='class-rand':
            g = self._generateLabelRandomBatches
        elif gen_type == 'class-itr':
            g = self._generateLabelIterativeBatches
        elif gen_type == 'rand':
            g = self._generateRandomBatches
        else:
            g = self._generateIterativeBatches
        return g(dataX, dataY, nb_batches)

    def _generateRandomBatches(self, dataX, dataY, nb_batches):
        'Generates random batches of samples'
        nb_samples = len(dataY)
        while 1:
          # Generate batches
          batchID = r.sample(range(0,nb_samples), self.batch_size)     
          X = [dataX[0][batchID], dataX[1][batchID], dataX[2][batchID]]
          y = pp.getMatrixLabels(self.unique_labels, dataY[batchID])
          yield X, y
    def _generateIterativeBatches(self, dataX, dataY, nb_batches):
        'Generates iterative batches of samples'
        nb_samples = len(dataY)
        while 1:
          # Generate batches
          for i in range(nb_batches):
              batchID = range(self.batch_size*i, min(nb_samples, self.batch_size*(i+1)))
#              X = [dataX[0][batchID], dataX[1][batchID], dataX[2][batchID]]
              X = [dataX[0][batchID], dataX[1][batchID]]
              y = pp.getMatrixLabels(self.unique_labels, dataY[batchID])
#              print(batchID)
              yield X, y
    def _generateLabelIterativeBatches(self, dataX, dataY, nb_batches):
        'Generates label iterative batches of samples'
        # Infinite loop
        while 1:
        
          # Generate batches
          for i in range(nb_batches):
              batchID = []
              for y in range(self.nb_classes):
                  cond = dataY==y
                  indicies = np.argwhere(cond)
                  indicies = np.squeeze(indicies)
                  for j in range(self.samples_per_label_in_batch):
                      index = (self.samples_per_label_in_batch*i + j) % len(indicies)
                      batchID.append(indicies[index])
#              X = [dataX[0][batchID], dataX[1][batchID], dataX[2][batchID]]
              X = [dataX[0][batchID], dataX[1][batchID]]
              y = pp.getMatrixLabels(self.unique_labels, dataY[batchID])
              yield X, y

    def _generateLabelRandomBatches(self, dataX, dataY, nb_batches):
        'Generates label iterative batches of samples'
        # Infinite loop
        while 1:
        
          # Generate batches
          for i in range(nb_batches):
              batchID = []
              for y in range(self.nb_classes):
                  cond = dataY==y
                  indicies = np.argwhere(cond)
                  indicies = np.squeeze(indicies)
                  indexes = r.sample(range(0,len(indicies)), 2)
                  batchID.extend(indicies[indexes])
              X = [dataX[0][batchID], dataX[1][batchID], dataX[2][batchID]]
              y = pp.getMatrixLabels(self.unique_labels, dataY[batchID])
              yield X, y

    def _configGenerators(self):
        if self.train_type.startswith( 'class' ):
            self.samples_per_label_in_batch = 2
            self.batch_size = self.nb_classes*self.samples_per_label_in_batch
            [_,C] = np.unique(self.trainY, return_counts=True)
            C[::-1].sort()
            maxC = C[self.cIdx]
            self.nb_train_batches = int(maxC/self.samples_per_label_in_batch)
            self.nb_val_batches = int(len(self.valY) / self.batch_size)
        else:
            self.batch_size = self.batch_size
            self.nb_train_batches = int(len(self.trainY) / self.batch_size)
            self.nb_val_batches = int(len(self.valY) / self.batch_size)
            
            
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
    
    def getInputs2Streams(self, prsBB, objBB, image):
        crops = pp.cropImageFromRel(prsBB, objBB, image)
        streams = pp.preprocessRel(crops['prsCrop'], crops['objCrop'], image, self.shape, self.mean)
        pairwise = self.getPairWiseStream(prsBB, objBB)
        streams['pairwise'] = pairwise
        return streams
    