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
import copy as cp
import random as r
import sys

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
    
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

## Load images ##
def loadImages(imagesID, imagesMeta, data_path):
    images = {}
    for imageID in imagesID:
        image = cv.imread(data_path+'/images/' + imagesMeta[imageID]['imageName'])
        if image is None:
            print(imageID)
        images[imageID] = image
    return images


def getBareBonesStats(labels):
    stats = {}
    for label in labels:
        obj = label['obj']; pred = label['pred']
        if obj not in stats:
            stats[obj] = {'total': 0}
        if pred not in stats[obj]:
            stats[obj][pred] =  (0,0)
#            stats[obj][pred+'conf'] =  0 
    return stats


def getLabelStats(imagesMeta, labels):
    counts = np.zeros(len(labels))
    stats = getBareBonesStats(labels)
    stats['total'] = 0
    stats['nb_samples'] = 0
    stats['nb_images'] = 0
    for imageID, imageMeta in imagesMeta.items():
        stats['nb_images'] += 1
        for relID, rel in imageMeta['rels'].items():
            stats['nb_samples'] += 1
            for idx in rel['labels']:
                name = labels[idx]
                (p,c) = stats[name['obj']][name['pred']]
                stats[name['obj']][name['pred']] = (p+1, c+min(1,len(rel['labels'])-1))
                stats[name['obj']]['total'] += 1
                stats['total'] += 1
                
                counts[idx] += 1
    return stats, counts

def idxs2labels(idxs, labels):
    reduced_labels = []
    for idx in idxs:
        reduced_labels.append(labels[idx])
    return reduced_labels

def _reduceData(imagesMeta, reduced_idxs):
    reduced_imagesMeta = {}
    for imageID, imageMeta in imagesMeta.items():
        new_rels = {}
        for relID, rel in imageMeta['rels'].items():
            new_labels = []
            for idx in rel['labels']:
                if idx in reduced_idxs:
                    new_labels.append(np.where(reduced_idxs==idx)[0][0])
            if len(new_labels) > 0:
                rel['labels'] = new_labels
                new_rels[relID] = rel
        if len(new_rels) > 0:
            reduced_imagesMeta[imageID] = {'imageName':imageMeta['imageName'], 'rels':new_rels}
    return reduced_imagesMeta    

def reduceTrainData(imagesMeta, counts, nb_classes):
    reduced_idxs = counts.argsort()[-nb_classes:][::-1]
    reduced_imagesMeta = _reduceData(imagesMeta, reduced_idxs)
    return reduced_imagesMeta, reduced_idxs

def reduceTestData(imagesMeta, reduced_idxs):
    return _reduceData(imagesMeta, reduced_idxs)



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

def createBackgroundBBs(imageMeta, nb_bgs, data_path):
    if 0 == nb_bgs:
        return []
    
    image = cv.imread(data_path + imageMeta['imageName'])
    corners = [[2, 0], [2, 1], [3, 1], [3, 0]]
    coors = []
    for relID, rel in imageMeta['rels'].items():
        xmin = rel['objBB']['xmin']
        xmax = rel['objBB']['xmax']
        ymin = rel['objBB']['ymin']
        ymax = rel['objBB']['ymax']
        coors.append([xmin, xmax, ymin, ymax])
        xmin = rel['prsBB']['xmin']
        xmax = rel['prsBB']['xmax']
        ymin = rel['prsBB']['ymin']
        ymax = rel['prsBB']['ymax']
        coors.append([xmin, xmax, ymin, ymax])
    coors = np.array(coors)
    bbs = []
    for corner in corners:
        coor_mask = np.array([1 for i in range(len(coors))])
        init_x = 0 if corner[1]==0 else image.shape[1]
        init_y = 0 if corner[0]==2 else image.shape[0]
        final_x = init_x; final_y = init_y
        for coor in range(10, 100, 5):
            x = coor if corner[1]==0 else image.shape[1] - coor
            y = coor if corner[0]==2 else image.shape[0] - coor
            idx = isPointInsideBoxes([x,y], coors[coor_mask.astype(np.bool)])
            if idx > 0:
                break
                
            final_x = x
            final_y = y
#        print('new corner', image.shape, final_x, final_y, init_x, init_y)    
        if abs(init_x-final_x) > 10 and abs(init_y-final_y) > 10:
            bb = {'xmin': min(init_x, final_x), 'xmax':max(init_x, final_x), 'ymin': min(init_y, final_y), 'ymax': max(init_y, final_y)}
            bbs.append(bb)
            
        if len(bbs) == nb_bgs*2:
            break
        
    while len(bbs) < nb_bgs*2:
        length = 25
        xmin, xmax = (0, length) if r.choice([0, 1]) else (image.shape[1]-length, image.shape[1])
        ymin, ymax = (0, length) if r.choice([0, 1]) else (image.shape[0]-length, image.shape[0])
        bb = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        bbs.append(bb)
                
        
    return bbs

def isPointInsideBoxes(point, boxes):
    for idx, box in enumerate(boxes):
        if point[0] > box[0] and point[0] < box[1] and point[1] > box[2] and point[1] < box[3]:
            return idx
    return 0

def concatXData(XMain, XSub):
    newXData = []
    if len(XMain) == 0:
        return XSub
    if len(XSub) == 0:
        return XMain
    for i in range(len(XMain)):
        newXData.append(np.append(XMain[i], XSub[i], axis=0))
    return newXData

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