# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')

#import extractTUHOIData as tuhoi
#import extractHICOData as hico
import utils
from generators import DataGenerator
from load_data import data

#from matplotlib import pyplot as plt
import cv2 as cv, numpy as np
import os
import sys
import time

#plt.close("all")

print('Loading data...')
# Load data
data = data(newDir=False)
cfg = data.cfg

    
if True:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  



if False:    
    i = 0
    start = time.time()
    print('Begin...')
    for sample in genTrain.begin():
#        utils.update_progress(i / genTrain.nb_batches)
        if i > genTrain.nb_batches:
            break
        i += 1
    end = time.time()
    print('End:', end - start)
    
    
if False:
    # Save labels in file
    annotations = hico.getUniqueLabels(cfg)
    labels = []
    for annot in annotations:
        obj = annot.nname; pred = annot.vname; pred_ing = annot.vname_ing
        label = {'obj': obj, 'pred': pred, 'pred_ing': pred_ing}
        labels.append(label)
        
    utils.save_dict(labels, cfg.part_data_path + 'HICO_labels')
    labels = utils.load_dict(cfg.part_data_path + 'HICO_labels')
    
    stats, counts = utils.getLabelStats(trainMeta, labels)
    reduced_trainMeta, reduced_idxs = utils.reduceTrainData(trainMeta, counts, 25)
    reduced_testMeta = utils.reduceTestData(testMeta, reduced_idxs)
    reduced_labels = utils.idxs2labels(reduced_idxs, labels)
    reduced_stats, reduced_counts = utils.getLabelStats(reduced_trainMeta, reduced_labels)

if False:
    # Test if all images can be loaded and cropped successfully
    i = 0
    c = 0.0
    end = len(testMeta)
    for imageID, metaData in testMeta.items():
        oldPath = cfg.part_data_path + 'HICO_images/test/' + metaData['imageID']
        image = cv.imread(oldPath)
        if i / end > c:
#            print(c)
            c += 0.01
#        print(str(c) + ': ' + str(imageID), end='')
        sys.stdout.write('\r' + str(c) + ': ' + str(imageID))
        sys.stdout.flush()
        if image is None:
            print(imageID)
        for relID, rel in metaData['rels'].items():
#            print(imageID, relID)
            relCrops = utils.cropImageFromRel(rel['prsBB'], rel['objBB'], image)
            relCrops = utils.preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, (227,227))
        i += 1


if False:
    # Debug problems with images/bounding boxes (like a wrongly rotated image)
    imageID = 'HICO_train2015_00027301.jpg'
    imageMeta = trainMeta[imageID]
    oldPath = cfg.part_data_path + 'HICO_images/train/' + imageMeta['imageID']
    image = cv.imread(oldPath)
    i = 0
    for relID, rel in metaData['rels'].items():
#            print(imageID, relID)
        relCrops = utils.cropImageFromRel(rel['prsBB'], rel['objBB'], image)
        relCrops = utils.preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, (227,227))
        i += 1

if False:
    # Plot images in range
    imagesID = list(trainMeta.keys())
    imagesID.sort()
    draw.drawImages(imagesID[26960:26969], trainMeta, labels, cfg.data_path+'_images/train/', False)

if True:
    # Test fast-generators by plotting data
    from fast_generators import DataGenerator
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    import matplotlib.pyplot as plt
    import draw
    i = 0
    idx = 0
    j = 0
    for sample in genTrain.begin():
#        utils.update_progress(j / len(data.trainMeta))
#        print(sample[1][idx])
        image = sample[0][0][idx]
        prsBB = sample[0][1][idx]
        objBB = sample[0][2][idx]
#        print(image.shape)
#        print(prsBB)
#        print(objBB)
#        draw.drawHOI(image, prsBB, objBB)
        i += 1
        if genTrain.nb_batches == i:
            break


if False:
    # Test generators by plotting data
    genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, data_type='train')
    i = 0
    idx = 25
    j = 0
    for sample in genTrain.begin():
        utils.update_progress(j / len(trainMeta))
#        print(len(sample))
        for ys in sample[1]:
            j += 1
        continue
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
    from extractHICOData import combineSimilarBBs
    # Check tu-ppmi images manually
#    oldStats, oldCounts = utils.getLabelStats(trainMeta, labels)
#    newStats, newCounts = utils.getLabelStats(newTrainMeta, labels)
    
    
    imagesMeta = trainMeta
    imagesID = list(imagesMeta.keys())
    imagesID.sort()
    i = 50
    n = 4
    imagesID = imagesID[i*n+2:i*n+n+2]
#    tmpMeta = {imageID: imageMeta for imageID, imageMeta in imagesMeta.items() if imageID in imagesID}
#    somethingelse = combineSimilarBBs(tmpMeta, labels, 0.4)
    print("i",i*n)
    draw.drawImages(imagesID, imagesMeta, labels, cfg.data_path +'images/train/', False)
    
    
if False:
    # Change tu-ppmi dict
    new_imagesMeta = {}
    for imageID, imageMeta in testMeta.items():
        new_rels = {}
        for relID, rel in imageMeta['rels'].items():
            new_rel = {'objBB':rel['objBB'], 'prsBB':rel['prsBB'], 'labels':rel['labels']}
            new_rels[relID] = new_rel
        new_imagesMeta[imageID] = {'imageName': imageMeta['imageID'], 'rels': new_rels}
        
    imagesID = list(new_imagesMeta.keys())
    imagesID.sort()
#    utils.save_dict(new_imagesMeta, cfg.data_path + 'test')
#    new_imagesMeta = utils.load_dict(cfg.data_path + 'test')
#    draw.drawImages(imagesID[50:59], new_imagesMeta, labels, cfg.data_path+'images/test/', False)
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
