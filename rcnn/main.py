# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""
import extractTUHOIData as tuhoi
import utils, draw
from model_trainer import model_trainer
from config import config
from models import AlexNet, PairWiseStream
from generators import DataGenerator
from methods import HO_RCNN, HO_RCNN_2

from matplotlib import pyplot as plt
import cv2 as cv, numpy as np
import copy as cp

from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from keras.layers import Add, Activation
from keras.models import Sequential, Model


plt.close("all")
unique_labels = tuhoi.getUniqueLabels()
nb_classes = len(unique_labels)
cfg = config(nb_classes=nb_classes)
#pp.save_obj(imagesMeta, 'TU_PPMI', url)
# Read data
if True:
    # Load data
#    imagesMeta, rawImagesMeta, garbage = tuhoi.extractMetaData()
#    print(garbage)
#    objects, allObjects = tuhoi.extractObjectData()
#    imagesMeta, imagesBadOnes = tuhoi.getBoundingBoxes(imagesMeta, objects, unique_labels)
    
    imagesMeta = utils.load_obj('TU_PPMI', cfg.data_path)
    imagesID = list(imagesMeta.keys())
    imagesID.sort()
    
    trainMeta, valMeta, testMeta = utils.splitData(imagesID, imagesMeta)
if True:
    # Create batch generators
    #class-itr
    genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, gen_type='itr')
    genVal = DataGenerator(imagesMeta=valMeta, cfg=cfg)
    genTest = DataGenerator(imagesMeta=testMeta, cfg=cfg)
    

if True:
    cfg.patience = 8
    cfg.epoch_begin = 0
    cfg.epoch_end = 1
    cfg.epoch_split = 10
    cfg.init_lr = 0.001
    cfg.task = 'multi-class'
    
    # Create model
    model = HO_RCNN(include_weights=True, nb_classes=cfg.nb_classes, task=cfg.task)
#    model = trainer.model
    # train model
    trainer = model_trainer(model=model, genTrain=genTrain, genVal=genVal, genTest=genTest, task=cfg.task)
    trainer.compileModel(wp=20, n_opt = 'adam')
    trainer.trainModel(cfg)
#    method.evaluateModel(gen.testX, gen.testY)
    
#    testYHat = method.evalYHat
#    testYMatrix = method.testYMatrix
#    testY = method.testY
    

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