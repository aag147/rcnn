# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:03:10 2018

@author: aag14
"""
import draw, utils
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

def getUniqueLabels(cfg):
    bbData = sio.loadmat(cfg.part_data_path + '../HICO/anno_bbox.mat', struct_as_record=False, squeeze_me=True)
    labels = bbData['list_action']
    return labels
    return [i for i in range(600)]

def extractMetaData(metaData):
    imagesMeta = {}
    for line in metaData:
        imageID = line.filename
        rels = {}
        relID = 0
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
            objBBs = np.array(rel.bboxobject).tolist()
            prsBBs = np.array(rel.bboxhuman).tolist()
            for subrel in subrels:
                idxsIntoAnnoLists = {'prsBB': subrel[0], 'objBB': subrel[1]}
                rels[relID] = {'labels': [hoiID]}
                for key, bbObject in {'objBB': objBBs, 'prsBB': prsBBs}.items():
                    ## BB coordinates
                    try:
                        idx = idxsIntoAnnoLists[key]
                        bbSrt   = bbObject[idx-1]
                    except TypeError:
                        bbSrt   = bbObject
                    xmin = bbSrt.x1; xmax = bbSrt.x2
                    ymin = bbSrt.y1; ymax = bbSrt.y2
                    bb = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
                    rels[relID][key] = bb
                relID += 1

        if rels:
            imagesMeta[imageID] = {'imageName': imageID, 'rels': rels}
            
    return imagesMeta


def combineSimilarBBs(imagesMeta):
    new_imagesMeta = {}
    for imageID, imageMeta in imagesMeta.items():
        data = {'prsBB':[], 'objBB':[], 'labels':[]}
        nb_rels = 0
        for relID, rel in imageMeta['rels'].items():
            data['prsBB'].append(rel['prsBB'])
            data['objBB'].append(rel['objBB'])
            data['labels'].extend(rel['labels'])
            nb_rels += 1
        
        for key in ['prsBB', 'objBB']:
            bbData = data[key]
            similars = np.array([i for i in range(nb_rels)], dtype=np.int)
            already_taken = []
            disabled = []
            while True:
                should_I_stay_or_should_I_go = 'go'
                for fstID, fstBB in enumerate(bbData[0:-1]):
                    if fstID in already_taken:
                        continue   
                    for secID, secBB in enumerate(bbData[fstID+1:]):
                        secID += fstID+1
                        if secID in disabled:
                            continue
                        if utils.get_iou(fstBB, secBB) > 0.4:
                            if similars[secID] != secID:
                                
                                similars[similars==fstID] = similars[secID]
                                already_taken.append(fstID)
                            else:
                                similars[similars==secID] = fstID
                                already_taken.append(secID)
                            should_I_stay_or_should_I_go = 'stay'
                if should_I_stay_or_should_I_go == 'go':
                    # converged
                    break
                new_bbData = [{} for i in range(len(bbData))]
                tmp_conn = []
                for sim in similars:
                    bb = bbData[sim]
                    if sim in tmp_conn:
                        meanBB = new_bbData[sim]
                        meanBB =  utils.meanBB(meanBB, bb)
                    else:
                        meanBB = bb
                    new_bbData[sim] = meanBB
                bbData = new_bbData
                disabled = already_taken
            data[key] = bbData
            data[key+'sims'] = similars
            
        tmp_rels = {}
        for relID in range(nb_rels):
            prsIdx = data['prsBBsims'][relID]
            objIdx = data['objBBsims'][relID]
            label = data['labels'][relID]
            if imageID == 'HICO_test2015_00000007.jpg':
                print(prsIdx, objIdx, label)
            if prsIdx not in tmp_rels:
                tmp_rels[prsIdx] = {}
            if objIdx not in tmp_rels[prsIdx]:
                tmp_rels[prsIdx][objIdx] = []
            tmp_rels[prsIdx][objIdx].append(label)
        
        rels = {}
        relID = 0
        for prsIdx, sub_rels in tmp_rels.items():
            for objIdx, labels in sub_rels.items():
                prsBB = data['prsBB'][prsIdx]
                objBB = data['objBB'][objIdx]
                rel = {'prsBB': prsBB, 'objBB': objBB, 'labels': labels}
                rels[relID] = rel
                relID += 1
        new_imagesMeta[imageID] = {'imageName': imageMeta['imageName'], 'rels': rels}
    return new_imagesMeta


                
if __name__ == "__main__":
    plt.close("all")
#    metaData = sio.loadmat(url + 'anno.mat', struct_as_record=False, squeeze_me=True)
#    bbData = sio.loadmat(url + 'anno_bbox.mat', struct_as_record=False, squeeze_me=True)
#    actions = bbData['list_action']
#    trainYMatrix = metaData['anno_train']
    bbDataTrain   = bbData['bbox_train']
    print("Extract meta data")
    tmpTrainMeta = extractMetaData(bbDataTrain)
    print("Combine similar BBs")
    newTrainMeta = combineSimilarBBs(tmpTrainMeta)
    newTrainMetaID = list(newTrainMeta.keys())
    newTrainMetaID.sort()
#    imagesID = imagesID[6490:7000]
#    images = pp.loadImages(imagesID, imagesMeta, url+"images/train2015/")
#    [dataXP, dataXB, dataY, dataMeta] = pp.getData(imagesID, imagesMeta, images, (224,244))
#    trainYMatrix = pp.getMatrixLabels(len(actions), dataY)
    utils.save_dict(newTrainMeta, url+'HICO_train')
#    sampleMeta = imagesMeta[imagesID[0]]
#    i = 0
#    pdata.drawImages(imagesID[i*9:(i+1)*9], imagesMeta, url+'images/train2015/', False)
