# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:12:41 2018

@author: aag14
"""

import metrics as m, \
       losses as l, \
       callbacks as cb


import numpy as np
import cv2 as cv
import random as r
import os

from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam

class model_trainer:
    def __init__(self, model, genTrain=None, genVal=None, genTest=None, task='multi-class'):
        self.task = task
        self.model = model
        self.genTrain = genTrain
        self.genVal = genVal
        self.eval = cb.EvaluateTest(genTest)
        self.log = cb.LogHistory()
        
    def saveLog(self, cfg):
#        multilabel = np.array(self.eval.multilabel)
#        mP = np.array(self.eval.mP)
#        mR = np.array(self.eval.mR)
#        F1 = np.array(self.eval.F1)
        train_loss = np.array(self.log.train_loss)
#        train_acc = np.array(self.log.train_acc)
        val_loss = np.array(self.log.val_loss)
#        val_acc = np.array(self.log.val_acc)
        
        for fid in range(100):
            if not os.path.exists(cfg.results_path + 'val_loss%d.out' % fid):
                np.savetxt(cfg.results_path + 'val_loss%d.out' % fid, val_loss, fmt='%.2f')
                np.savetxt(cfg.results_path + 'train_loss%d.out' % fid, train_loss, fmt='%.2f')
                break
        return val_loss, train_loss
    
    def compileModel(self, cfg):
        if cfg.optimizer == 'adam':
            opt = Adam(lr=0.001, decay=0.0)
        else:
            opt = SGD(lr = 0.001, momentum = 0.9, decay = 0.0, nesterov=False)
        if self.task == 'multi-label':
            loss = l.weigthed_binary_crossentropy(cfg.wp,1)
        else:
            loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        
    def trainModel(self, cfg):
        callbacks = [self.log, \
                     cb.MyEarlyStopping(cfg), \
                     cb.MyModelCheckpoint(cfg), \
                     cb.MyLearningRateScheduler(cfg), \
                     cb.PrintCallBack()]
        
        if cfg.include_eval:
            callbacks.append(self.eval)
        
        
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