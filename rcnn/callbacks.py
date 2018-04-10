# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import metrics as m

import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
from keras import backend as K


def MyEarlyStopping(patience = ''):
    return EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')


def MyModelCheckpoint(key = '', cfg):
    path = cfg.weights_path + key + 'weights.{epoch:02d}-{val_loss:.2f}.h5'
    return ModelCheckpoint(path+filename, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

def MyLearningRateScheduler(epoch_split = 5, init_lr = 0.001):
   def step_decay(epoch):
      if epoch >= epoch_split:
          return init_lr / 10.0
      else:
          return init_lr
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
   def __init__(self, gen):
      self.gen = gen
      self.nb_samples = gen.nb_samples
      self.nb_batches = gen.nb_batches
      self.nb_classes = gen.nb_classes
      self.batch_size = gen.batch_size
      
      self.F1 = []
      self.mP = []
      self.mR = []
       
      self.multilabel = []
      self.zeroAcc = []
   def on_train_begin(self, logs=None):
       return
		
   def on_epoch_end(self, epoch, logs={}):
      self.evalYHat = np.zeros([self.nb_samples, self.nb_classes])
      self.y = np.zeros([self.nb_samples, self.nb_classes])
      iterGen = self.gen.begin()
      for i in range(self.nb_batches):
          batch, y = next(iterGen)
#          print(batch[0].shape)
#          print(y.shape)
          y_hat = self.model.predict_on_batch(x=batch)
          s_idx = i * self.batch_size
          f_idx = min(self.nb_samples,s_idx+self.batch_size)
          self.evalYHat[s_idx:f_idx, :] = y_hat
          self.y[s_idx:f_idx, :] = y
#      print(self.evalYHat)
#      print(self.evalYHat.shape)

      accs, mP, mR, F1 = m.computeMultiLabelLoss(self.y, self.evalYHat)
      self.multilabel.append(accs)
      self.mP.append(mP)
      self.mR.append(mR)
      self.F1.append(F1)
      
   def on_epoch_begin(self, epoch, logs={}):
       if len(self.F1) > 0:
           print('test_f1:', self.F1[-1])
		
			
   def on_train_end(self, logs=None):
      return