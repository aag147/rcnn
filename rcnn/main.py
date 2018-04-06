# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""
import extractTUHOIData as tuhoi
import preprocess as pp
import plotData as pdata
from matplotlib import pyplot as plt
import cv2 as cv, numpy as np
import copy as cp
from HO_RCNN import HO_RCNN
from models import AlexNet, PairWiseStream
from generators import DataGenerator

from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from keras.layers import Add, Activation
from keras.models import Sequential, Model


plt.close("all")
#pp.save_obj(imagesMeta, 'TU_PPMI', url)
# Read data
if False:
    # Load data (should be done by generator working on big data)
    url = tuhoi.getURL()
    unique_labels = tuhoi.getUniqueLabels()
    rawImagesMeta, imagesMeta = tuhoi.extractMetaData()
    objects, allObjects = tuhoi.extractObjectData()
    imagesMeta, imagesBadOnes = tuhoi.getBoundingBoxes(imagesMeta, objects, unique_labels)
    imagesMeta = pp.load_obj('TU_PPMI', url)
    imagesID = list(imagesMeta.keys())
    imagesID.sort()
    imagesID = imagesID
    
    images = pp.loadImages(imagesID, imagesMeta, url+"images/")
    [trainID, valID, testID] = pp.splitData(imagesID)
    
if True:
    # Create batch generators
    #class-itr
    gen = DataGenerator(train_type='class-itr', val_type='itr')
    gen.setDataVariables(allID=imagesID, trainID=trainID, valID=valID, testID=testID, imagesMeta=imagesMeta, \
                            images=images, unique_labels=unique_labels)
    

if True:
    # Create and run model
    method = HO_RCNN(unique_classes=unique_labels, class_type='multi-class')
    method.createModel()
    #method.setModel(methodW.model)
    method.compileModel(wp=1)
    method.trainModel(start_epoch=0, final_epoch=15, generator=gen)
#    method.evaluateModel(gen.testX, gen.testY)
    
#    testYHat = method.evalYHat
#    testYMatrix = method.testYMatrix
#    testY = method.testY


if False:
    mean  = [103.939, 116.779, 123.68]
    shape = (227, 227)
    urlWeights = 'C:/Users/aag14/Documents/Skole/Speciale/Weights/'
    
    [trainID, testID] = pp.splitData(imagesID)
    [trainPrsX, trainObjX, trainY, trainMeta] = pp.getData(trainID, imagesMeta, images, shape)
    trainParX = method.getDataPairWiseStream(trainID, imagesMeta)
    testParX = method.getDataPairWiseStream(testID, imagesMeta)
    [testPrsX, testObjX, testY, testMeta] = pp.getData(testID, imagesMeta, images, shape)
    trainYMatrix = pp.getMatrixLabels(unique_labels, trainY)
    modelPrs = AlexNet(urlWeights+"alexnet_weights.h5", len(unique_labels), include='fc')
    modelObj = AlexNet(urlWeights+"alexnet_weights.h5", len(unique_labels), include='fc')
    modelPar = PairWiseStream(nb_classes = len(unique_labels), include='fc')
                       
    mergedOut = Add()([modelPrs.output, modelObj.output, modelPar.output])
    mergedOut = Activation("softmax",name="softmax")(mergedOut)
    model = Model(input=[modelPrs.input, modelObj.input, modelPar.input], output=mergedOut)
    
    def step_decay(epoch):
        if epoch > 3:
            return 0.0001
        else:
            return 0.001
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    lrate = LearningRateScheduler(step_decay)
    sgd = SGD(lr = 0.001, momentum = 0.8, decay = 0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit([trainPrsX, trainObjX, trainParX],trainYMatrix,epochs=12,batch_size=64, callbacks=[earlyStopping, lrate], validation_split=0.20)
    
    y_hat_p = model.predict(x=[testPrsX,testObjX,testParX])
    

if False:
    i = 0
    idx = 31
    for sample in gen.generateTrain():
        print(np.argmax(sample[1][idx]))
        win = sample[0][2][idx]
        prs = sample[0][0][idx].transpose([1,2,0])
        obj = sample[0][1][idx].transpose([1,2,0])
        f, spl = plt.subplots(2,2)
        spl = spl.ravel()
        spl[0].imshow(win[0], cmap=plt.cm.gray)
        spl[1].imshow(win[1], cmap=plt.cm.gray)
        spl[2].imshow((prs - np.min(prs)) / np.max(prs))
        spl[3].imshow((obj - np.min(obj)) / np.max(obj))
        i += 1
        if i == 5:
            break

if False:
    imagesMeta = pp.load_obj('TU_PPMI', url)
    stats = tuhoi.getLabelStats(imagesMeta)
    i = 100
    print("i",i*9)
    pdata.drawImages(imagesID[i*9:i*9+9], imagesMeta, url+'images/', imagesBadOnes)

#imageID = imagesID[1]
#imageMeta = imagesMeta[imageID]
#rel = imageMeta['rels'][0]
#image = images[imageID]
#f, spl = plt.subplots(2,2)
#spl = spl.ravel()
#spl[0].imshow(res['pairwise'][0], cmap=plt.cm.gray)
#spl[1].imshow(res['pairwise'][1], cmap=plt.cm.gray)
#spl[2].imshow(image)
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