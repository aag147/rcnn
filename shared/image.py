# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:45:39 2018

@author: aag14
"""

import cv2 as cv
import numpy as np
import utils

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
def getXData(imagesID, imagesMeta, data_path, shape):
    dataX = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        image = cv.imread(data_path + imageMeta['imageName'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imageClean = preprocessImage(image, shape)
        dataX.append(imageClean)
    dataX = np.array(dataX)
    return dataX    

def getX2Data(imagesID, imagesMeta, data_path, shape):
    dataXP = []
    dataXB = []
#    print(imagesID, imagesMeta)
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        image = cv.imread(data_path + imageMeta['imageName'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        for relID, rel in imageMeta['rels'].items():
            relCrops = cropImageFromRel(rel['prsBB'], rel['objBB'], image)
            relCrops = preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, shape)
            dataXP.append(relCrops['prsCrop'])
            dataXB.append(relCrops['objCrop'])
    dataXP = np.array(dataXP)
    dataXB = np.array(dataXB)
    return [dataXP, dataXB]

## Resize and normalize image ##
def preprocessImage(image, shape):
    image = cv.resize(image, shape).astype(np.float32)
#    image = (image - np.mean(image)) / np.std(image)
#    print(np.max(image))
#    print('im', np.min(image), np.max(image), np.isfinite(image).all())
    image = (image - np.min(image)) / np.max(image)
    image = image.transpose([2,0,1])
    return image

def preprocessRel(prsCrop, objCrop, image, shape):
    prsCrop = preprocessImage(prsCrop, shape)
    objCrop = preprocessImage(objCrop, shape)
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
