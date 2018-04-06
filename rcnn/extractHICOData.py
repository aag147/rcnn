# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:03:10 2018

@author: aag14
"""
import newplotData as pdata
import newpreprocess as pp
import csv
url = 'C:/Users/aag14/Documents/Skole/Speciale/HICO/'


import glob
import os
import cv2 as cv
import xml.etree.ElementTree as ET  
import scipy.io as sio
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

def extractMetaData(metaData, actions):
    imagesMeta = {}
    for line in metaData:
        imageID = line.filename
        rels = {}
        bbLists = {'prsBB':{}, 'objBB':{}}
        tmp_rels = {}
        #print(imageID)
        try:
            line.hoi[0]
        except TypeError:
            line.hoi = [line.hoi]
        for rel in line.hoi:
            subrels = np.array(rel.connection).tolist()
            #print(subrels)
            if rel.invis:
                continue
            if not isinstance(subrels[0], list):
                subrels = [subrels]
            hoiID = rel.id - 1
            objName = actions[hoiID].nname
            pred    = actions[hoiID].vname
            objBBs = np.array(rel.bboxobject).tolist()
            prsBBs = np.array(rel.bboxhuman).tolist()
            for subrel in subrels:
                idxsIntoPredLists = {'prsBB': subrel[0], 'objBB': subrel[1]}
                idxsIntoBBLists = {'prsBB': None, 'objIdx': None}
                for key, bbObject in {'objBB': objBBs, 'prsBB': prsBBs}.items():
                    ## BB coordinates
                    try:
                        idx = idxsIntoPredLists[key]
                        bbSrt   = bbObject[idx-1]
                    except TypeError:
                        bbSrt   = bbObject
                    xmin = bbSrt.x1; xmax = bbSrt.x2
                    ymin = bbSrt.y1; ymax = bbSrt.y2
                    bb = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
                    idx = None
                    for BBID, exisBB in bbLists[key].items():
                        if imageID == 'HICO_train2015_00006593.jpg':
                            print(imageID, key, pdata.get_iou(exisBB, bb))
                        if pdata.get_iou(exisBB, bb) > 0.4:
                            idx = BBID
                            bb = pdata.meanBB(bb, exisBB)
                            bbLists[key][idx] = bb
                            break
                    if idx is None:
                        idx = len(bbLists[key])
                        bbLists[key][idx] = bb
                    idxsIntoBBLists[key] = idx
                if idxsIntoBBLists['prsBB'] not in tmp_rels.keys():
                    tmp_rels[idxsIntoBBLists['prsBB']] = {}
                if idxsIntoBBLists['objBB'] not in tmp_rels[idxsIntoBBLists['prsBB']].keys():
                    tmp_rels[idxsIntoBBLists['prsBB']][idxsIntoBBLists['objBB']] = {'names':[], 'labels':[]}
                tmp_rels[idxsIntoBBLists['prsBB']][idxsIntoBBLists['objBB']]['names'].append({'obj':objName, 'pred':pred})
                tmp_rels[idxsIntoBBLists['prsBB']][idxsIntoBBLists['objBB']]['labels'].append(hoiID)
        relIdx = 0
        for prsID, sub_tmp_rels in tmp_rels.items():
            for objID, intacts in sub_tmp_rels.items():
                rels[relIdx] = {'prsBB':bbLists['prsBB'][prsID], \
                                'objBB':bbLists['objBB'][objID], \
                                'names': intacts['names'], \
                                'labels': intacts['labels']}                    
                relIdx += 1
        if rels:
            imagesMeta[imageID] = {'imageID': imageID, 'rels': rels}
    return imagesMeta


if __name__ == "__main__":
    plt.close("all")
#    metaData = sio.loadmat(url + 'anno.mat', struct_as_record=False, squeeze_me=True)
#    bbData = sio.loadmat(url + 'anno_bbox.mat', struct_as_record=False, squeeze_me=True)
#    actions = bbData['list_action']
#    trainYMatrix = metaData['anno_train']
#    bbDataTrain   = bbData['bbox_train']
#    imagesMeta = extractMetaData(bbDataTrain, actions)
    imagesID = list(imagesMeta.keys())
    imagesID.sort()
    imagesID = imagesID[6490:7000]
    images = pp.loadImages(imagesID, imagesMeta, url+"images/train2015/")
    [dataXP, dataXB, dataY, dataMeta] = pp.getData(imagesID, imagesMeta, images, (224,244))
    trainYMatrix = pp.getMatrixLabels(len(actions), dataY)
    
    sampleMeta = imagesMeta[imagesID[0]]
    i = 0
    pdata.drawImages(imagesID[i*9:(i+1)*9], imagesMeta, url+'images/train2015/', False)
