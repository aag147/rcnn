# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:45:39 2018

@author: aag14
"""

import cv2 as cv
import numpy as np
import utils
import sys

#%% Y DATA
def getYData(imagesID, imagesMeta, GTMeta, cfg):
    dataLabels = []
    dataBBs    = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        for relID, rel in imageMeta['rels'].items():
            labels, prsGT, objGT = getGTDataRel(rel, GTMeta[imageID], cfg)
            dataLabels.append(labels)
            dataBBs.append(prsGT + objGT)
    dataLabels = (dataLabels)
    dataBBs = np.array(dataBBs)
    dataLabels = utils.getMatrixLabels(cfg.nb_classes, dataLabels)
    return dataLabels, dataBBs


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


#%% X DATA
## Get model ready data ##
def getXData(imagesID, imagesMeta, data_path, cfg):
    dataX = []
    dataH = []
    dataO = []
    IDs   = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        image = cv.imread(data_path + imageMeta['imageName'])
#        sys.stdout.write('\r' + str(imageID))
#        sys.stdout.flush()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imageClean = preprocessImage(image, cfg)
        
        scaleX = imageClean.shape[1] / float(image.shape[1])
        scaleY = imageClean.shape[0] / float(image.shape[0])
        scales = [scaleX, scaleY]
#        print(imageClean.shape, image.shape, scales)
        
        for relID, rel in imageMeta['rels'].items():
            h, o = getDataFromRel(rel['prsBB'], rel['objBB'], scales, imageClean.shape, cfg)
            dataX.append(imageClean)
            dataH.append(h)
            dataO.append(o)
            IDs.append(imageID)
    dataX = np.array(dataX)
    dataH = np.array(dataH)
    dataO = np.array(dataO)
    IDs = np.array(IDs)
    
#    dataH = np.expand_dims(dataH, axis=1)
#    dataO = np.expand_dims(dataO, axis=1)
    
    return [dataX, dataH, dataO], IDs

def getX2Data(imagesID, imagesMeta, data_path, cfg):
    dataXP = []
    dataXB = []
#    print(imagesID, imagesMeta)
    for imageID in imagesID:
#        sys.stdout.write('\r' + str(imageID))
#        sys.stdout.flush()
        imageMeta = imagesMeta[imageID]
        image = cv.imread(data_path + imageMeta['imageName'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        for relID, rel in imageMeta['rels'].items():
            relCrops = cropImageFromRel(rel['prsBB'], rel['objBB'], image)
            relCrops = preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, cfg)
            dataXP.append(relCrops['prsCrop'])
            dataXB.append(relCrops['objCrop'])
    dataXP = np.array(dataXP)
    dataXB = np.array(dataXB)
    return [dataXP, dataXB]

## Resize and normalize image ##
def preprocessImage(image, cfg):
    image = cv.resize(image, cfg.shape).astype(np.float32)
#    image = (image - np.mean(image)) / np.std(image)
#    print(np.max(image))
#    print('im', np.min(image), np.max(image), np.isfinite(image).all())
    image = (image - np.min(image)) / np.max(image)
    image = image.transpose(cfg.order_of_dims)
    return image

def preprocessRel(prsCrop, objCrop, image, cfg):
    prsCrop = preprocessImage(prsCrop, cfg)
    objCrop = preprocessImage(objCrop, cfg)
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

## Get data bbs from rel ##
def getDataFromBBOld(bb, scales, cfg):
    x = (bb['xmin'] * scales[0] * (1/cfg.img_out_reduction[0]))
    y = (bb['ymin'] * scales[1] * (1/cfg.img_out_reduction[1]))
    w = (bb['xmax'] * scales[0] * (1/cfg.img_out_reduction[0])  - x)
    h = (bb['ymax'] * scales[1] * (1/cfg.img_out_reduction[1]) - y)
    return [x, y, w, h]

def getDataFromBB(bb, scales, shape, cfg):
    xmin = bb['xmin'] * scales[0] #* (1/cfg.img_out_reduction[0]))
    ymin = bb['ymin'] * scales[1] #* (1/cfg.img_out_reduction[1]))
    xmax = bb['xmax'] * scales[0] #* (1/cfg.img_out_reduction[0]))
    ymax = bb['ymax'] * scales[1] #* (1/cfg.img_out_reduction[1]))
    
    xmin = xmin / float(shape[1]); xmax = xmax / float(shape[1])
    ymin = ymin / float(shape[0]); ymax = ymax / float(shape[0])
    return [ymin, xmin, ymax, xmax]
    
def getDataFromRel(prsBB, objBB, scales, shape, cfg):
    dataH = getDataFromBB(prsBB, scales, shape, cfg)
    dataO = getDataFromBB(objBB, scales, shape, cfg)
    return dataH, dataO


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

def getDataPairWiseStream(imagesID, imagesMeta, cfg):
    dataPar = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        for relID, rel in imageMeta['rels'].items():
            relWin = _getPairWiseStream(rel['prsBB'], rel['objBB'], cfg)
            dataPar.append(relWin)
    dataPar = np.array(dataPar)
    dataPar = dataPar.transpose(cfg.par_order_of_dims)
    return dataPar
