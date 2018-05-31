# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:03:10 2018

@author: aag14
"""

import sys 
sys.path.append('../shared/')
sys.path.append('../../')

url = 'C:/Users/aag14/Documents/Skole/Speciale/HICO/'

import utils
from config import basic_config
from config_helper import set_config
import csv


import glob
import os
import cv2 as cv
import xml.etree.ElementTree as ET  
import scipy.io as sio
import matplotlib.gridspec as gridspec

import numpy as np

def getUniqueLabels(cfg):
    bbData = sio.loadmat(cfg.part_data_path + '../HICO/anno_bbox.mat', struct_as_record=False, squeeze_me=True)
    labels = bbData['list_action']
    return labels
    return [i for i in range(600)]

def extractMetaData(metaData, labels):
    imagesMeta = {}
    mlk = 0
    for line in metaData:
        imageID = line.filename
        
        objs = []
        
        rels = []
        #print(imageID)
        try:
            line.hoi[0]
        except TypeError:
            line.hoi = [line.hoi]
            
        for rel in line.hoi:
            
            if rel.invis:
                mlk += 1
                continue
            
            hoiID = rel.id - 1
            
            subrels = np.array(rel.connection).tolist()
            if not isinstance(subrels[0], list):
                subrels = [subrels]
            
            objStructs = np.array(rel.bboxobject).tolist()
            prsStructs = np.array(rel.bboxhuman).tolist()
            try:
                objStructs[0]
            except TypeError:
                objStructs = [objStructs]
            try:
                prsStructs[0]
            except TypeError:
                prsStructs = [prsStructs]    
                
            objLabel = labels[hoiID]['obj']
            
            for objSt in objStructs:
                xmin = objSt.x1; xmax = objSt.x2
                ymin = objSt.y1; ymax = objSt.y2
                objBB = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
                objBB['label'] = objLabel
                rels.append(objBB)
            
            for prsSt in prsStructs:
                xmin = prsSt.x1; xmax = prsSt.x2
                ymin = prsSt.y1; ymax = prsSt.y2
                prsBB = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
                prsBB['label'] = 'person'
                rels.append(prsBB)



        if rels:
            imagesMeta[imageID.split('.')[0]] = {'imageName': imageID, 'objects': rels}
#    print(mlk)
    return imagesMeta



                
if __name__ == "__main__":
#    metaData = sio.loadmat(url + 'anno.mat', struct_as_record=False, squeeze_me=True)
    bbData = sio.loadmat(url + 'anno_bbox.mat', struct_as_record=False, squeeze_me=True)
#    actions = bbData['list_action']
#    trainYMatrix = metaData['anno_train']
    bbDataTrain   = bbData['bbox_train']
    cfg = basic_config()
    cfg = set_config(cfg)
    cfg.dataset = 'HICO'
    cfg.get_data_path()
    cfg.get_results_paths()
    labels = utils.load_dict(cfg.data_path + 'labels')
    print("Extract meta data")
    tmpTrainMeta = extractMetaData(bbDataTrain, labels)

    utils.save_dict(tmpTrainMeta, url+'train_objs')
