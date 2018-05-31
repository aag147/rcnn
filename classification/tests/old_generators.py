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
      gt_label, _, _ = self.getYData(self.dataID, self.imagesMeta, self.GTMeta, self.cfg)
      self.nb_images = len(self.dataID)
      self.nb_samples = gt_label.shape[1]
      if self.nb_batches is None:
          self.nb_batches = m.ceil(self.nb_samples / self.batch_size)
      
      
      
    def begin(self):
        'Generates batches of samples'

        g = self._generateIterativeBatches
        return g()
    
    def _generateBatchFromIDs(self, batchID):
        batchID = [self.dataID[idx] for idx in batchID]
#        print(batchID)
        [dataXP, dataXB] = self.getX2Data(batchID, self.imagesMeta, self.images_path, self.cfg)
        dataXW = image.getDataPairWiseStream(batchID, self.imagesMeta, self.cfg)
        X = [dataXP, dataXB, dataXW]
        X = [X[i] for i in range(len(X)) if self.inputs[i]]
        y, _, _ = self.getYData(batchID, self.imagesMeta, self.GTMeta, self.cfg)
        y = y[0]
        return X, y

    #%% Different forms of generators

    def _generateIterativeBatches(self):
        'Generates iterative batches of samples'
        
        
        while 1:
          imageIdx = 1
          hoiinimageidx = 0
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
#                      hoiinimageidx = 0
                      
                  if (len(imageY) - hoiinimageidx) + len(y) >= self.batch_size:
                      hoiinimageidx = hoiinimageidx + len(imageY) - ((len(imageY) + len(y)) - self.batch_size)
                      f_idx = hoiinimageidx
                  else:
                     hoiinimageidx = 0
                      
                  imageXCut = utils.spliceXData(imageX, s_idx, f_idx)
                  X = utils.concatXData(X, imageXCut)
                  y.extend(imageY[s_idx:f_idx, :])
#                  y.extend([self.dataID[imageIdx] for i in range(s_idx, f_idx)])
                  if len(y) == self.batch_size:
                      break
              imageIdx += 1
              y = np.array(y)
              yield X, y
    

    def getX2Data(self, imagesID, imagesMeta, data_path, cfg):
        dataXP = []
        dataXB = []
    #    print(imagesID, imagesMeta)
        for imageID in imagesID:
    #        sys.stdout.write('\r' + str(imageID))
    #        sys.stdout.flush()
            imageMeta = imagesMeta[imageID]
    #        print(data_path + imageMeta['imageName'])
            image = cv.imread(data_path + imageMeta['imageName'])
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            for relID, rel in imageMeta['rels'].items():
                relCrops = self.cropImageFromRel(rel['prsBB'], rel['objBB'], image)
                relCrops = self.preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, cfg)
                dataXP.append(relCrops['prsCrop'])
                dataXB.append(relCrops['objCrop'])
        dataXP = np.array(dataXP)
        dataXB = np.array(dataXB)
        return [dataXP, dataXB]          
              
    def preprocessCrop(self, image, cfg):
        image = cv.resize(image, cfg.shape).astype(np.float32)
    #    image = (image - np.mean(image)) / np.std(image)
    #    print(np.max(image))
    #    print('im', np.min(image), np.max(image), np.isfinite(image).all())
        image = (image - np.min(image)) / np.max(image)
        image = image.transpose(cfg.order_of_dims)
        return image
    
    def preprocessRel(self, prsCrop, objCrop, image, cfg):
        prsCrop = self.preprocessCrop(prsCrop, cfg)
        objCrop = self.preprocessCrop(objCrop, cfg)
        return {'prsCrop': prsCrop, 'objCrop': objCrop}
        
    def cropImageFromBB(self, bb, image):
        # Single image, half relation
        xmin = bb['xmin']; xmax = bb['xmax']
        ymin = bb['ymin']; ymax = bb['ymax']
    #    print(image.shape, [ymin, ymax, xmin, xmax])
        crop = image[ymin:ymax, xmin:xmax, :]
        return crop
    
    def cropImageFromRel(self, prsBB, objBB, image):
        # Single image, single relation
        prsCrop = self.cropImageFromBB(prsBB, image)
        objCrop = self.cropImageFromBB(objBB, image)
        crops = {'prsCrop': prsCrop, 'objCrop': objCrop}
        return crops
              
    def getYData(self,imagesID, imagesMeta, GTMeta, cfg):
        dataLabels = []
        dataHumanBBs    = []
        dataObjectBBs    = []
        for imageID in imagesID:
            imageMeta = imagesMeta[imageID]
            relsY = []
            relsH = []
            relsO = []
            for relID, rel in imageMeta['rels'].items():
                labels = self._getGTDataRel(rel, GTMeta[imageID], cfg)
                relsY.append(labels)
                relsH.append(None)
                relsO.append(None)
            relsY = utils.getMatrixLabels(cfg.nb_classes, relsY)
            dataLabels.append(relsY)
            dataHumanBBs.append(relsH)
            dataObjectBBs.append(relsO)
        dataLabels = np.vstack(dataLabels)
        dataLabels = np.expand_dims(dataLabels, axis=0)
        dataHumanBBs = np.array(dataHumanBBs)
        dataObjectBBs = np.array(dataObjectBBs)
        return dataLabels, dataHumanBBs, dataObjectBBs
    
    def _getGTDataRel(self, myRel, GTMeta, cfg):
        labels = []
        for relID, relGT in GTMeta['rels'].items():
            prsIoU = utils.get_iou(myRel['prsBB'], relGT['prsBB'])
            objIoU = utils.get_iou(myRel['objBB'], relGT['objBB'])
            if prsIoU > cfg.minIoU and objIoU > cfg.minIoU:
                labels.append(relGT['label'])
        return labels
    
