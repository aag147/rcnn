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


def MyModelCheckpointBest(cfg):
    path = cfg.my_weights_path + 'model-best.h5'
    return ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def MyModelCheckpointInterval(cfg):
    path = cfg.my_weights_path + 'model-{epoch:03d}.h5'
    return ModelCheckpoint(path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=cfg.checkpoint_interval)

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
      
class SaveLog2File(Callback):
   def __init__(self, cfg):
      self.cfg = cfg
      f= open(cfg.my_results_path + "history.txt","a")
      f.close()
   def on_epoch_end(self, epoch, logs=None):
       val_loss = logs.get('val_loss') if type(logs.get('val_loss')) is float else 0.0
       val_acc  = logs.get('val_acc') if type(logs.get('val_acc')) is float else 0.0
       val_loss = logs.get('rpn_out_class_loss') if logs.get('rpn_out_class_loss') is not None else 0.0
       val_acc  = logs.get('rpn_out_regress_loss') if logs.get('rpn_out_regress_loss') is not None else 0.0
       val_loss = logs.get('det_out_class_loss') if logs.get('det_out_class_loss') is not None else 0.0
       val_acc  = logs.get('det_out_regress_loss') if logs.get('det_out_regress_loss') is not None else 0.0
       train_acc  = logs.get('acc') if type(logs.get('acc')) is float else 0.0
       newline = '%.03d, %.4f, %.4f, %.4f, %.4f\n' % \
         (epoch, logs.get('loss'), train_acc, val_loss, val_acc)
       with open(self.cfg.my_results_path + "history.txt", 'a') as file:
           file.write(newline)
		
class LogHistory(Callback):
   def __init__(self):
      self.hist = m.LogHistory()

   def on_epoch_end(self, epoch, logs={}):
      self.hist.newEpoch(logs)
	
class EvaluateTest(Callback):
   def __init__(self, gen, f, cfg):
      self.gen = gen
      self.f = f  
      self.epochs = []
      self.cfg = cfg
      f= open(cfg.my_results_path + "evals.txt","w+")
      f.close()
   def on_train_begin(self, logs=None):
       return
		
   def on_epoch_end(self, epoch, logs={}):
       results = self.f(self.model, self.gen)
       self.epochs.append(results)
       with open(self.cfg.my_results_path + "evals.txt", 'a') as file:
           newline = '%.03d, %.4f, %.03d' % \
             (epoch, results.F1, results.nb_zeros)
           file.write(newline)
      
   def on_epoch_begin(self, epoch, logs={}):
       if epoch > 0:
           print('test_f1:', self.epochs[-1].F1)
		
			
   def on_train_end(self, logs=None):
      return
