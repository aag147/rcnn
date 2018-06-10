# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:12:41 2018

@author: aag14
"""

import metrics as m, \
       losses as l, \
       callbacks as cb, \
       utils


import numpy as np
import cv2 as cv
import random as r
import copy as cp
import os

from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam

class model_trainer:
    def __init__(self, model, genTrain=None, genVal=None, genTest=None, task='multi-class'):
        self.task = task
        self.model = model
        self.genTrain = genTrain
        self.genVal = genVal
        self.genTest = genTest
        self.log = cb.LogHistory()
    
    def compileModel(self, cfg):
        if cfg.optimizer == 'adam':
            print('Optimzer: adam')
            opt = Adam(lr=0.001)
        else:
            opt = SGD(lr = 0.001, momentum = 0.9, decay = 0.0, nesterov=False)
        if self.task == 'multi-label':
            loss = l.weigthed_binary_crossentropy(cfg.wp,1)
        else:
            loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=opt, metrics=['categorical_accuracy'])

        
    def trainModel(self, cfg):
        callbacks = [self.log, \
                     cb.MyModelCheckpointInterval(cfg), \
                     cb.MyLearningRateScheduler(cfg), \
                     cb.SaveLog2File(cfg), \
                     cb.PrintCallBack()]
        
        if cfg.include_eval:
            callbacks.append(cb.EvaluateTest(self.genTest, m.EvalResults, cfg))
            
            
            
        if cfg.include_validation:
            callbacks.append(cb.MyEarlyStopping(cfg))
            callbacks.append(cb.MyModelCheckpointBest(cfg))
            self.model.fit_generator(generator = self.genTrain.begin(), \
                    steps_per_epoch = self.genTrain.nb_batches, \
                    validation_data = self.genVal.begin(), \
                    validation_steps = self.genVal.nb_batches, \
                    epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)
        else:
            self.model.fit_generator(generator = self.genTrain.begin(), \
                    steps_per_epoch = self.genTrain.nb_batches, \
                    epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)


    def evaluateModel(self, gen):
        return m.EvalResults(self.model, gen)

    def saveHistory(self, cfg):
        res = cp.copy(self.eval.epochs)
        log = cp.copy(self.log.hist)
        log.train_loss = np.array(log.train_loss)
        log.train_acc = np.array(log.train_acc)
        log.val_loss = np.array(log.val_loss)
        log.val_acc = np.array(log.val_acc)
        
        for fid in range(100):
            path = cfg.my_results_path
            if not os.path.exists(path + 'log%d.pkl' % fid):
                utils.save_obj(log, path + 'log%d' % fid)
                utils.save_obj(res, path + 'res%d' % fid)
                break
    
    def saveModel(self, cfg):
        for fid in range(100):
            weight_path = cfg.my_weights_path + 'weights-thend%d.h5' % fid
            model_path = cfg.my_weights_path + 'model-thend%d.h5' % fid
            if not os.path.exists(weight_path):
                self.model.save_weights(weight_path)
                self.model.save(model_path)
                break