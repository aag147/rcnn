# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:12:41 2018

@author: aag14
"""

import metrics as m, \
       losses as l, \
       callbacks as cb

from models import AlexNet, PairWiseStream, getWeightsURL

import numpy as np
import cv2 as cv
import random as r

from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam
from keras.layers import Add, Activation
from keras.models import Sequential, Model
from keras import backend as K

class model_trainer:
    def __init__(self, model, genTrain=None, genVal=None, genTest=None, task='multi-class'):
        self.task = task
        self.model = model
        self.genTrain = genTrain
        self.genVal = genVal
        self.eval = cb.EvaluateTest(genTest)
        self.log = cb.LogHistory()
        
    def getEvals(self):
#        multilabel = np.array(self.eval.multilabel)
#        mP = np.array(self.eval.mP)
#        mR = np.array(self.eval.mR)
        F1 = np.array(self.eval.F1)
#        train_loss = np.array(self.log.train_loss)
#        train_acc = np.array(self.log.train_acc)
        val_loss = np.array(self.log.val_loss)
#        val_acc = np.array(self.log.val_acc)
        return val_loss, F1
    
    def compileModel(self, wp=2, n_opt = 'sgd'):
        if n_opt == 'adam':
            opt = Adam(lr=0.001, decay=0.0)
        else:
            opt = SGD(lr = 0.001, momentum = 0.9, decay = 0.0, nesterov=False)
        if self.task == 'multi-label':
            loss = l.weigthed_binary_crossentropy(wp,1)
        else:
            loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        
    def trainModel(self, cfg):
        callbacks = [self.log, \
                     self.eval, \
                     cb.MyEarlyStopping(patience=cfg.patience), \
                     cb.MyModelCheckpoint(key=cfg.modelnamekey), \
                     cb.MyLearningRateScheduler(epoch_split=cfg.epoch_split, init_lr=cfg.init_lr), \
                     cb.PrintCallBack()]
        
        
        self.model.fit_generator(generator = self.genTrain.begin(), \
                    steps_per_epoch = self.genTrain.nb_batches, \
                    validation_data = self.genVal.begin(), \
                    validation_steps = self.genVal.nb_batches, \
                    epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)

    def evaluateModel(self, genTest):
        if X is None:
            X = self.testX
            y = self.testY
        EvaluateTest()
        self.evalYHat = self.model.predict(x=X)
        return m.computeMultiLabelLoss(y, self.evalYHat)