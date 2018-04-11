# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('..')
sys.path.append('rcnn/')

import extractTUHOIData as tuhoi
import utils, draw
from config import config
from config_helper import set_config
from generators import DataGenerator

from matplotlib import pyplot as plt
import cv2 as cv, numpy as np
import os


plt.close("all")
cfg = config()
cfg = set_config(cfg)

# Read data
if True:
    # Load data 
    
#    imagesMeta, rawImagesMeta, garbage = tuhoi.extractMetaData()
#    print(garbage)
#    objects, allObjects = tuhoi.extractObjectData()
#    imagesMeta, imagesBadOnes = tuhoi.getBoundingBoxes(imagesMeta, objects, unique_labels)
#    
#    trainMeta = utils.load_obj(cfg.data_path+'_train')
#    testMeta = utils.load_obj(cfg.data_path+'_test')
#    imagesID = list(imagesMeta.keys())
#    imagesID.sort()
    
    trainMeta = utils.load_obj(cfg.data_path+'_train')
    testMeta = utils.load_obj(cfg.data_path+'_test') 
#    trainMeta, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
    
    
if True:
    for imageID, metaData in trainMeta.items():
        oldPath = cfg.part_data_path + 'TU_PPMI_images/train/' + metaData['imageID']
        image = cv.imread(oldPath)
        for relID, rel in metaData['rels'].items():
            #print(imageID, relID)
            relCrops = utils.cropImageFromRel(rel['prsBB'], rel['objBB'], image)
            relCrops = utils.preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, (227,227))
        if image is None:
            print(imageID)
#        os.rename(oldPath, newPath)

    
if False:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, gen_type=cfg.train_type)
    genVal = DataGenerator(imagesMeta=valMeta, cfg=cfg, gen_type=cfg.val_type)
    genTest = DataGenerator(imagesMeta=testMeta, cfg=cfg, gen_type=cfg.test_type)

if False:
    i = 0
    idx = 0
    for sample in genTrain.begin():
        print(np.argmax(sample[1][idx]))
        win = sample[0][2][idx]
        prs = sample[0][0][idx].transpose([1,2,0])
        obj = sample[0][1][idx].transpose([1,2,0])
        f, spl = plt.subplots(2,2)
        spl = spl.ravel()
        spl[0].imshow(win[0], cmap=plt.cm.gray)
        spl[1].imshow(win[1], cmap=plt.cm.gray)
        spl[2].imshow(prs)
        spl[3].imshow(obj)
        i += 1
        if i == 5:
            break

if False:
    imagesMeta = utils.load_obj('TU_PPMI', url)
    stats = tuhoi.getLabelStats(imagesMeta)
    i = 100
    print("i",i*9)
    draw.drawImages(imagesID[i*9:i*9+9], imagesMeta, url+'images/', imagesBadOnes)

#
#imageID = trainMeta[1]
#X, y = next(genTrain)
#imageMeta = imagesMeta[imageID]
#rel = imageMeta['rels'][0]
#image = images[imageID]
#f, spl = plt.subplots(2,2)
#spl = spl.ravel()
#spl[0].imshow(X[2][0], cmap=plt.cm.gray)
#spl[1].imshow(X[2][1], cmap=plt.cm.gray)
#spl[2].imshow(X[0])
#spl[3].imshow(X[1])
#
#print(np.unique(res['pairwise'][0]))
#print(np.unique(res['pairwise'][1]))

#mean = [103.939, 116.779, 123.68]
#shape = (224, 224)
#images = pp.loadImages(imagesID, imagesMeta, url+"images/")
#
#imagesCrops = pp.cropImagesFromRels(imagesID, imagesMeta, images)
#resizedImagesCrops = pp.preprocessCrops(imagesCrops, shape, mean)
#[trainID, testID] = pp.splitData(imagesID)
#
#pdata.drawImages([imageID], imagesMeta, url+'images/', False)
#
#pdata.drawCrops(imagesID, imagesMeta, imagesCrops, images)