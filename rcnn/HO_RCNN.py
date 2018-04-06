# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:12:41 2018

@author: aag14
"""

import preprocess as pp
import numpy as np
import cv2 as cv
import metrics as m
import losses as l
import random as r

from models import AlexNet, PairWiseStream, getWeightsURL
import callbacks as cb

from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD
from keras.layers import Add, Activation
from keras.models import Sequential, Model
from keras import backend as K

class HO_RCNN:
    def __init__(self, unique_classes=None, class_type='multi-class'):
        self.weightsURL = getWeightsURL()
        self.nb_classes = len(unique_classes)
        self.class_type = class_type

    def createModel(self):
        modelPrs = AlexNet(self.weightsURL+"alexnet_weights.h5", self.nb_classes, include='fc')
        modelObj = AlexNet(self.weightsURL+"alexnet_weights.h5", self.nb_classes, include='fc')
#        modelPar = PairWiseStream(nb_classes = self.nb_classes, include='fc')
                           
#        mergedOut = Add()([modelPrs.output, modelObj.output, modelPar.output])
        mergedOut = Add()([modelPrs.output, modelObj.output])
        if self.class_type == 'multi-label':
            mergedOut = Activation("sigmoid",name="predictions")(mergedOut)
        else:
            mergedOut = Activation("softmax",name="predictions")(mergedOut)
#        model = Model(input=[modelPrs.input, modelObj.input, modelPar.input], output=mergedOut)
        model = Model(input=[modelPrs.input, modelObj.input], output=mergedOut)
       
        self.model = model
        
    def setModel(self, model):
        self.model = model
    
    def compileModel(self, wp=2):        
        sgd = SGD(lr = 0.001, momentum = 0.9, decay = 0.0, nesterov=False)
        if self.class_type == 'multi-label':
            loss = l.weigthed_binary_crossentropy(wp,1)
        else:
            loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

        
    def trainModel(self, start_epoch=0, final_epoch=1, generator=None):
        self.log = cb.LogHistory()
        self.evalTest = cb.EvaluateTest(generator.testX[0:2], generator.testYMatrix, self.class_type)
        callbacks = [self.log, \
                     self.evalTest, \
                     EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='auto'), \
                     cb.MyLearningRateScheduler(), \
                     cb.PrintCallBack()]
        
        
        self.model.fit_generator(generator = generator.generateTrain(), \
                    steps_per_epoch = generator.nb_train_batches, \
                    validation_data = generator.generateVal(), \
                    validation_steps = generator.nb_val_batches, \
                    epochs = final_epoch, initial_epoch=start_epoch, callbacks=callbacks)

    def evaluateModel(self, X=None, y=None):
        if X is None:
            X = self.testX
            y = self.testY
        self.evalYHat = self.model.predict(x=X)
        return m.computeMultiLabelLoss(y, self.evalYHat)