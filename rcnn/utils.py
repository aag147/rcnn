# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:45:39 2018

@author: aag14
"""
import cv2 as cv
import numpy as np
import sklearn.model_selection as skmodel
import json
import glob, os
import pickle


def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_dict(obj, path):
    with open(path + '.json', 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)

def load_dict(path):
    with open(path + '.json', 'r') as f:
        return json.load(f)

## Load images ##
def loadImages(imagesID, imagesMeta, data_path):
    images = {}
    for imageID in imagesID:
        image = cv.imread(data_path+'_images/' + imagesMeta[imageID]['imageID'])
        if image is None:
            print(imageID)
        images[imageID] = image
    return images

## Split data (IDs) ##
def splitData(imagesID, imagesMeta):
    [trainID, testID] = skmodel.train_test_split(imagesID, test_size=0.2)
#    [trainID, valID] = skmodel.train_test_split(trainID, test_size=0.2)
    trainMeta = {key:imagesMeta[key] for key in trainID}
#    valMeta = {key:imagesMeta[key] for key in valID}
    testMeta = {key:imagesMeta[key] for key in testID}
    return trainMeta, testMeta


def spliceXData(XData, s_idx, f_idx):
    newXData = []
    for i in range(len(XData)):
        newXData.append(XData[i][s_idx:f_idx])
    return newXData

def concatXData(XMain, XSub):
    newXData = []
    if len(XMain) == 0:
        return XSub
    for i in range(len(XMain)):
        newXData.append(np.append(XMain[i], XSub[i], axis=0))
    return newXData

## Get model ready data ##
def getXData(imagesID, imagesMeta, data_path, shape):
    dataX = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        image = cv.imread(data_path + imageMeta['imageID'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imageClean = preprocessImage(image, shape)
        dataX.append(imageClean)
    dataX = np.array(dataX)
    return dataX    

def getX2Data(imagesID, imagesMeta, data_path, shape):
    dataXP = []
    dataXB = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        image = cv.imread(data_path + imageMeta['imageID'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        for relID, rel in imageMeta['rels'].items():
            #print(imageID, relID)
            relCrops = cropImageFromRel(rel['prsBB'], rel['objBB'], image)
            relCrops = preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, shape)
            dataXP.append(relCrops['prsCrop'])
            dataXB.append(relCrops['objCrop'])
    dataXP = np.array(dataXP)
    dataXB = np.array(dataXB)
    return [dataXP, dataXB]

def getYData(imagesID, imagesMeta, nb_classes):
    dataLabels = []
    dataBBs    = []
    for imageID in imagesID:
        imageMeta = imagesMeta[imageID]
        for relID, rel in imageMeta['rels'].items():
            labels, prsGT, objGT = getGTDataRel(rel)
            dataLabels.append(labels)
            dataBBs.append(prsGT + objGT)
    dataLabels = (dataLabels)
    dataBBs = np.array(dataBBs)
    dataLabels = getMatrixLabels(nb_classes, dataLabels)
    return dataLabels, dataBBs


def getUniqueClasses(cfg):
    if cfg.dataset == 'TU_PPMI':
        from extractTUHOIData import getUniqueLabels
    else:
        from extractHICOData import getUniqueLabels
    unique_labels = getUniqueLabels(cfg)
    return len(unique_labels)


def getGTData(bb):
    xmin = bb['xmin']; xmax = bb['xmax']
    ymin = bb['ymin']; ymax = bb['ymax']
    return [xmin, xmax, ymin, ymax]

def getGTDataRel(rel):
    labels = rel['labels']
    prsGT = getGTData(rel['prsBB'])
    objGT = getGTData(rel['objBB'])
    return labels, prsGT, objGT

## Resize and normalize image ##
def preprocessImage(image, shape):
    image = cv.resize(image, shape).astype(np.float32)
#    image = (image - np.mean(image)) / np.std(image)
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

## Convert label int to list ##
def getMatrixLabels(nb_classes, Y):
    YMatrix = np.zeros((len(Y), nb_classes))
    sIdx = 0
    for y in Y:
        for clIdx in y:
            YMatrix[sIdx][clIdx] = 1
        sIdx += 1
    return YMatrix

def getVectorLabels(YMatrix):
    Y = np.zeros(YMatrix.shape[0], dtype=int)
    sIdx = 0
    for sIdx in range(len(Y)):
        y = np.argmax(YMatrix[sIdx,:])
        Y[sIdx] = y
    return Y

def deleteFillerFiles(keepers, path, ext):
    i = 0
    for filepath in glob.iglob(path +'*.' + ext):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        if filename not in keepers:
            os.remove(filepath)
            i += 1
    return i

def meanBB(bb1, bb2):
    xmin = int((bb1['xmin'] + bb2['xmin']) / 2)
    xmax = int((bb1['xmax'] + bb2['xmax']) / 2)
    ymin = int((bb1['ymin'] + bb2['ymin']) / 2)
    ymax = int((bb1['ymax'] + bb2['ymax']) / 2)
    return {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
    
def get_iou(bb1, bb2, include_union = True):
    """
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area
    if include_union:
        iou = iou / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
    else:
        iou = iou / float(bb1_area)
    return iou