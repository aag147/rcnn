# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:45:39 2018

@author: aag14
"""

import cv2 as cv
import numpy as np
import utils
import sys
import random
import copy as cp

#%% Y DATA
def getYData(imagesID, imagesMeta, GTMeta, cfg, batch_size=None):
    dataLabels = []
    dataHumanBBs    = []
    dataObjectBBs    = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        relsY = []
        relsH = []
        relsO = []
        i = 0
        for relID, rel in imageMeta['rels'].items():
            labels, prsGT, objGT = getGTDataRel(rel, GTMeta[imageID], cfg)
            relsY.append(labels)
            relsH.append(prsGT)
            relsO.append(objGT)
            i += 1
            if batch_size is not None and i >= batch_size:
                break
        relsY = utils.getMatrixLabels(cfg.nb_classes, relsY)
        dataLabels.append(relsY)
        dataHumanBBs.append(relsH)
        dataObjectBBs.append(relsO)
    dataLabels = np.vstack(dataLabels)
    dataLabels = np.expand_dims(dataLabels, axis=0)
    dataHumanBBs = np.array(dataHumanBBs)
    dataObjectBBs = np.array(dataObjectBBs)
    return dataLabels, dataHumanBBs, dataObjectBBs


def getGTLabels(myRel, GTMeta, cfg):
    labels = []
    for relID, relGT in GTMeta['rels'].items():
        prsIoU = utils.get_iou(myRel['prsBB'], relGT['prsBB'])
        objIoU = utils.get_iou(myRel['objBB'], relGT['objBB'])
        if prsIoU > cfg.minIoU and objIoU > cfg.minIoU:
            labels.append(relGT['label'])
    return labels

def getGTData(bb):
    xmin = bb['xmin']; xmax = bb['xmax']
    ymin = bb['ymin']; ymax = bb['ymax']
    return [xmin, xmax, ymin, ymax]

def getGTDataRel(rel, GTMeta, cfg):
    labels = getGTLabels(rel, GTMeta, cfg)
    prsGT = getGTData(rel['prsBB'])
    objGT = getGTData(rel['objBB'])
    return labels, prsGT, objGT


#%% Fast DATA
def getImage(imageMeta, data_path, cfg):
    image = cv.imread(data_path + imageMeta['imageName'])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imageClean, scale, padds = preprocessImage(image, cfg)
    return imageClean, scale

def getBoxes(imageMeta, cfg):
    image = cv.imread(data_path + imageMeta['imageName'])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imageClean, scale, padds = preprocessImage(image, cfg)
    return imageClean, scale

#%% X DATA
## Get model ready data ##
def getXData(imagesMeta, imagesID, images_path, cfg, batchIdx):
    dataX = []
    dataH = []
    dataO = []
    IDs   = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        img = cv.imread(images_path + imageMeta['imageName'])
#        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        assert(img is not None)
        assert(img.shape[0] > 10)
        assert(img.shape[1] > 10)
        assert(img.shape[2] == 3)
        
        imgRedux, scale = preprocessImage(img, cfg)
        tmpH = []
        tmpO = []
        for relID, rel in imageMeta['rels'].items():
            h, o = getDataFromRel(rel['prsBB'], rel['objBB'], scale, [0,0], imgRedux.shape, cfg)
            tmpH.append([batchIdx] + h)
            tmpO.append([batchIdx] + o)
        tmpH = np.array(tmpH)
        tmpO = np.array(tmpO)
        if len(tmpH)>0 and cfg.flip_image and random.choice([True, False]):
            imgRedux = np.fliplr(imgRedux)
            tmp = cp.copy(tmpH[:,2])
            tmpH[:,2] = 1.0 -  tmpH[:,4]
            tmpH[:,4] = 1.0 -  tmp
            tmp = cp.copy(tmpO[:,2])
            tmpO[:,2] = 1.0 -  tmpO[:,4]
            tmpO[:,4] = 1.0 -  tmp
         
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

def getX2Data(imageMeta, data_path, cfg):
    dataXP = []
    dataXB = []

    image = cv.imread(data_path + imageMeta['imageName'])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    imageMeta['flip'] = False
    if cfg.flip_image and random.choice([True, False]):
        imageMeta['flip'] = True
        imageMeta['shape'] = image.shape
        imageMeta = utils.flipMeta(imageMeta)
        image = np.fliplr(image)
    
    for relID, rel in imageMeta['rels'].items():
        relCrops = cropImageFromRel(rel['prsBB'], rel['objBB'], image)
        relCrops = preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, cfg)
        
        prsCrop = relCrops['prsCrop']
        objCrop = relCrops['objCrop']
        
        dataXP.append(prsCrop)
        dataXB.append(objCrop)
            
    dataXP = np.array(dataXP)
    dataXB = np.array(dataXB)
    return [dataXP, dataXB]

#%% Fast rcnn
## Resize and normalize image ##
def preprocessImage(img, cfg):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32, copy=False)
    shape = img.shape
    if shape[0] < shape[1]:
        newHeight = cfg.mindim
        scale = float(newHeight) / shape[0]
        newWidth = int(shape[1] * scale)
        if newWidth > cfg.maxdim:
            newWidth = cfg.maxdim
            scale = newWidth / shape[1]
            newHeight = shape[0] * scale
    else:
        newWidth = cfg.mindim
        scale = float(newWidth) / shape[1]
        newHeight = int(shape[0] * scale)
        if newHeight > cfg.maxdim:
            newHeight = cfg.maxdim
            scale = newHeight / shape[0]
            newWidth = shape[1] * scale


    newWidth = round(newWidth)
    newHeight = round(newHeight)

    scaleWidth = float(newWidth) / img.shape[1]
    scaleHeight = float(newHeight) / img.shape[0]
    scales = [scaleHeight, scaleWidth]
    # Rescale
    img = cv.resize(img, (newWidth, newHeight)).astype(np.float32)
    # Normalize
    if False:# or cfg.use_channel_mean:
        img -= cfg.img_channel_mean
        img /= cfg.img_scaling_factor
    else:
#        img = (img - np.min(img)) / np.max(img)
        img /= 127.5
        img -= 1.0
    # Transpose
    img = img.transpose(cfg.order_of_dims)
    return img, scales

    
def getDataFromBB(bb, scales, padds, shape, cfg):
    xmin = bb['xmin'] * scales[1] + padds[1] #* (1/cfg.img_out_reduction[0]))
    ymin = bb['ymin'] * scales[0] + padds[0] #* (1/cfg.img_out_reduction[1]))
    xmax = bb['xmax'] * scales[1] + padds[1] #* (1/cfg.img_out_reduction[0]))
    ymax = bb['ymax'] * scales[0] + padds[0] #* (1/cfg.img_out_reduction[1]))
    
    xmin = xmin / float(shape[1]); xmax = xmax / float(shape[1])
    ymin = ymin / float(shape[0]); ymax = ymax / float(shape[0])
    return [ymin, xmin, ymax, xmax]
    
def getDataFromRel(prsBB, objBB, scales, padds, shape, cfg):
    dataH = getDataFromBB(prsBB, scales, padds, shape, cfg)
    dataO = getDataFromBB(objBB, scales, padds, shape, cfg)
    return dataH, dataO

#%% rcnn
def preprocessCrop(img, cfg):
    img = img.astype(np.float32, copy=False)
    img = cv.resize(img, cfg.shape).astype(np.float32)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#    image = (image - np.mean(image)) / np.std(image)
#    print(np.max(image))
#    print('im', np.min(image), np.max(image), np.isfinite(image).all())
    if False:# or cfg.use_channel_mean:
        img -= cfg.img_channel_mean
        img /= cfg.img_scaling_factor
    else:
        img = (img - np.min(img)) / np.max(img)
#        img /= 127.5
#        img -= 1.0
    img = img.transpose(cfg.order_of_dims)
    return img

def preprocessRel(prsCrop, objCrop, image, cfg):
    prsCrop = preprocessCrop(prsCrop, cfg)
    objCrop = preprocessCrop(objCrop, cfg)
    return {'prsCrop': prsCrop, 'objCrop': objCrop}

## Crop image ##
def cropImageFromBB(bb, image):
    # Single image, half relation
    xmin = bb['xmin']; xmax = bb['xmax']
    ymin = bb['ymin']; ymax = bb['ymax']
#    print(image.shape, [ymin, ymax, xmin, xmax])
    crop = image[ymin:ymax, xmin:xmax, :]
    return crop

def cropImageFromRel(prsBB, objBB, image):
    # Single image, single relation
    prsCrop = cropImageFromBB(prsBB, image)
    objCrop = cropImageFromBB(objBB, image)
    crops = {'prsCrop': prsCrop, 'objCrop': objCrop}
    return crops


#%% Special third stream data extraction
def _getSinglePairWiseStream(thisBB, thatBB, width, height, newWidth, newHeight, cfg):
    xmin = max(0, thisBB['xmin'] - thatBB['xmin'])
    xmax = width - max(0, thatBB['xmax'] - thisBB['xmax'])
    ymin = max(0, thisBB['ymin'] - thatBB['ymin'])
    ymax = height - max(0, thatBB['ymax'] - thisBB['ymax'])
    
    attWin = np.zeros([height,width])
    attWin[ymin:ymax, xmin:xmax] = 1
    attWin = cv.resize(attWin, (newWidth, newHeight), interpolation = cv.INTER_NEAREST)
    attWin = attWin.astype(np.int)

    xPad = int(abs(newWidth - cfg.winShape[0]) / 2)
    yPad = int(abs(newHeight - cfg.winShape[0]) / 2)
    attWinPad = np.zeros(cfg.winShape).astype(np.int)
#        print(attWin.shape, attWinPad.shape, xPad, yPad)
#        print(height, width, newHeight, newWidth)
    attWinPad[yPad:yPad+newHeight, xPad:xPad+newWidth] = attWin
    return attWinPad

def _getPairWiseStream(prsBB, objBB, cfg):
    width = max(prsBB['xmax'], objBB['xmax']) - min(prsBB['xmin'], objBB['xmin'])
    height = max(prsBB['ymax'], objBB['ymax']) - min(prsBB['ymin'], objBB['ymin'])
    if width > height:
        newWidth = cfg.winShape[0]
        apr = newWidth / width
        newHeight = int(height*apr) 
    else:
        newHeight = cfg.winShape[0]
        apr = newHeight / height
        newWidth = int(width*apr)
        
    prsWin = _getSinglePairWiseStream(prsBB, objBB, width, height, newWidth, newHeight, cfg)
    objWin = _getSinglePairWiseStream(objBB, prsBB, width, height, newWidth, newHeight, cfg)
    
    return [prsWin, objWin]

def getDataPairWiseStream(imageMeta, cfg):
    dataPar = []
    for relID, rel in imageMeta['rels'].items():
        relWin = _getPairWiseStream(rel['prsBB'], rel['objBB'], cfg)
        dataPar.append(relWin)
    
    dataPar = np.array(dataPar)
    dataPar = dataPar.transpose(cfg.par_order_of_dims)
    return dataPar
