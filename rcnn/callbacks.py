# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import metrics as m
import os

import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
from keras import backend as K


def MyEarlyStopping(cfg):
    return EarlyStopping(monitor='val_loss', patience=cfg.patience, verbose=0, mode='auto')


def MyModelCheckpoint(cfg):
    for fid in range(100):
        if not os.path.exists(cfg.weights_path + cfg.modelnamekey + 'weights%d/' % fid):
            path = cfg.weights_path + cfg.modelnamekey + cfg.dataset + '_weights%d/' % fid
            os.mkdir(path)
            break
    path = path + 'weights.{epoch:02d}-{val_loss:.2f}.h5'
    return ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

def MyLearningRateScheduler(cfg):
   epoch_splits = cfg.epoch_splits
   init_lr = cfg.init_lr
   def step_decay(epoch):
      for idx in range(len(epoch_splits)):
          if epoch < epoch_splits[idx]:
              return init_lr / pow(10.0, idx)
      return init_lr / pow(10.0, idx+1)
   return LearningRateScheduler(step_decay)
	
class PrintCallBack(Callback):
   def on_epoch_begin(self, epoch, logs=None):
      print("learning rate", K.eval(self.model.optimizer.lr))
		
class LogHistory(Callback):
   def __init__(self):
      self.hist = m.LogHistory()

   def on_epoch_end(self, epoch, logs={}):
      self.hist.newEpoch(logs)
	
class EvaluateTest(Callback):
   def __init__(self, gen, f):
      self.gen = gen
      self.f = f  
      self.epochs = []
   def on_train_begin(self, logs=None):
       return
		
   def on_epoch_end(self, epoch, logs={}):
       results = self.f(self.model, self.gen)
       self.epochs.append(results)
      
   def on_epoch_begin(self, epoch, logs={}):
       if len(self.epochs) > 0:
           print('test_f1:', self.epochs[-1].F1)
		
			
   def on_train_end(self, logs=None):
      return