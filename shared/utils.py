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
import shutil


def getPascalObjects(nb_classes):
    objs = ['person', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',\
     'dog', 'horse', 'motorcycle', 'potted plant', 'sheep', 'couch', 'train', 'tv']
    
    return objs[0:nb_classes]


#%% Save / load
def load_hist(path):
    hist = []
    f = open(path,"r")
    for line in f.readlines():
        line = line.split(', ')
        hist.append(line[1:])
    f.close()
    hist = np.array(hist).astype(dtype=float)
    return hist

def save_obj_nooverwrite(obj, path, protocol = 2):
    for fid in range(100):
        path = path + '%d' % fid
        if not os.path.exists(path+'.pkl'):
           with open(path+'.pkl', 'wb') as f:
               pickle.dump(obj, f, protocol)
               return

def save_obj(obj, path, protocol = 2):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_dict(obj, path):
    with open(path + '.JSON', 'w') as f:
        json.dump(obj, f, sort_keys=True, indent=4)

def load_dict(path):
    with open(path + '.JSON', 'r') as f:
        return json.load(f)

def saveConfig(cfg):
   obj = cp.copy(cfg)
   obj = vars(obj)
   obj['train_cfg'] = vars(obj['train_cfg'])
   obj['val_cfg'] = vars(obj['val_cfg'])
   obj['test_cfg'] = vars(obj['test_cfg'])
#   obj['img_channel_mean'] = obj['img_channel_mean'].tolist()
   obj['wp'] = obj['wp'] if type(obj['wp']) is int else obj['wp'].tolist()
   for fid in range(100):
        path = cfg.my_results_path
        if not os.path.exists(path + 'cfg%d.json' % fid):
            save_dict(obj, path + 'cfg%d' % fid)
            break 
        
def saveSplit(cfg, trainID, valID):
    path = cfg.my_results_path
    save_obj(trainID, path + 'trainIDs')
    save_obj(valID, path + 'valIDs')
    
    
def moveData(from_path, to_path):
    shutil.copytree(from_path, to_path)
    
def removeData(to_path):
    shutil.rmtree(to_path)


# %% Process loading
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
    
    
def update_progress_new(itr, total, imageName):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    progress = float(itr) / total
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
    if isinstance(progress, float):
        status = imageName
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}/{2} - {3}".format( "#"*block + "-"*(barLength-block), itr, total, status)
    sys.stdout.write(text)
    sys.stdout.flush()



# %% Statistics
def getBareBonesStats(labels):
    stats = {}
    for label in labels:
        obj = label['obj']; pred = label['pred']
        if obj not in stats:
            stats[obj] = {'total': 0}
        if pred not in stats[obj]:
            stats[obj][pred] =  0
#            stats[obj][pred+'conf'] =  0 
    return stats


def getLabelStats(imagesMeta, labels):
    counts = np.zeros(len(labels))
    stats = getBareBonesStats(labels)
    stats['total'] = 0
    stats['totalx'] = 0
    stats['nb_samples'] = 0
    stats['nb_images'] = 0
    for imageID, imageMeta in imagesMeta.items():
        stats['nb_images'] += 1
        
        stats['nb_samples'] += len(imageMeta['rels'])
        stats['total'] += len(imageMeta['rels'])
        stats['totalx'] += min(32,len(imageMeta['rels']))
        
        for relID, rel in imageMeta['rels'].items():
            if 'labels' in rel:
                continue
            
            idx = rel['label']
            name = labels[idx]
            stats[name['obj']][name['pred']] += 1
            stats[name['obj']]['total'] += 1
                
            counts[idx] += 1
    return stats, counts

# %% Reduce data
def reduceData(imagesMeta, reduced_idxs):
    reduced_imagesMeta = {}
    for imageID, imageMeta in imagesMeta.items():
        new_rels = {}
        for relID, rel in imageMeta['rels'].items():
            if 'labels' in rel:
                # Proposals
                new_labels = []
                for idx in rel['labels']:
                    if idx in reduced_idxs:
                        new_labels.append(np.where(reduced_idxs==idx)[0][0])
                if len(new_labels) > 0:
                    rel['labels'] = new_labels
                    new_rels[relID] = rel
            else:
                # GT
                idx = rel['label']
                if idx in reduced_idxs:
                    new_label = np.where(reduced_idxs==idx)[0][0]
                    rel['label'] = new_label
                    new_rels[relID] = rel
        if len(new_rels) > 0:
            reduced_imagesMeta[imageID] = {'imageName':imageMeta['imageName'], 'rels':new_rels}
    return reduced_imagesMeta   

def idxs2labels(idxs, labels):
    reduced_labels = []
    for idx in idxs:
        reduced_labels.append(labels[idx])
    return reduced_labels 


def getReducedIdxs(counts, nb_classes, labels):
    if nb_classes >= 50:
        reduced_idxs = counts.argsort()[-nb_classes:][::-1]
    else:
        objs = getPascalObjects(nb_classes)
        reduced_idxs = []
        for idx, label in enumerate(labels):
            if label['obj'] in objs:
                reduced_idxs.append(idx)
        reduced_idxs = np.array(reduced_idxs)    
    return reduced_idxs

# %% Split and concat data
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
    if len(XSub) == 0:
        return XMain
    for i in range(len(XMain)):
        newXData.append(np.append(XMain[i], XSub[i], axis=0))
    return newXData

# %% Random background bbs
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


# %% Y Matrix to Y Vector, and opposite
def getMatrixLabels(nb_classes, Y):
    YMatrix = np.zeros((len(Y), nb_classes))
    sIdx = 0
    for y in Y:
        if y is not np.ndarray:
            y = [y]
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

def getMatrixDeltas(nb_classes, deltas, labels):
    YMatrix = np.zeros((len(labels), (nb_classes-1)*4))
    sIdx = 0
    for sidx in range(len(labels)):
        ds = deltas[sidx]
        l = labels[sidx]
        if l == 0:
            continue
        YMatrix[sidx, (l-1)*4:l*4] = ds
        sIdx += 1
    return YMatrix    


#%% IOU 
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
#    print('areas', bb1_area, bb2_area)
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
