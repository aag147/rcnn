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
        if not os.path.exists(cfg.weights_path + cfg.modelnamekey + '%d/' % fid):
            path = cfg.weights_path + cfg.modelnamekey + '%d/' % fid
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
      self.train_loss = []
      self.train_acc = []
      self.val_loss = []
      self.val_acc = []
   def on_train_begin(self, logs={}):
      return

   def on_epoch_end(self, epoch, logs={}):
      self.train_loss.append(logs.get('loss'))
      self.train_acc.append(logs.get('acc'))
      self.val_loss.append(logs.get('val_loss'))
      self.val_acc.append(logs.get('val_acc'))
	
class EvaluateTest(Callback):
   def __init__(self, gen, f):
      self.gen = gen
      self.f = f
      
      self.F1 = []
      self.mP = []
      self.mR = []
       
      self.multilabel = []
      self.zeroAcc = []
   def on_train_begin(self, logs=None):
       return
		
   def on_epoch_end(self, epoch, logs={}):
      accs, mP, mR, F1, nb_zeros = self.f(self.model, self.gen)
      self.multilabel.append(accs)
      self.mP.append(mP)
      self.mR.append(mR)
      self.F1.append(F1)
      self.zeroAcc.append(nb_zeros)
      
   def on_epoch_begin(self, epoch, logs={}):
       if len(self.F1) > 0:
           print('test_f1:', self.F1[-1])
		
			
   def on_train_end(self, logs=None):
      return