# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import metrics as m

import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras import backend as K


def MyLearningRateScheduler(epoch_split = 5):
   def step_decay(epoch):
      if epoch > epoch_split:
          return 0.0001
      else:
          return 0.001
   return LearningRateScheduler(step_decay)
	
class PrintCallBack(Callback):
   def on_epoch_begin(self, epoch, logs=None):
      print("learning rate", K.eval(self.model.optimizer.lr))
		
class LogHistory(Callback):
   def on_train_begin(self, logs={}):
      self.train_loss = []
      self.train_acc = []
      self.val_loss = []
      self.val_acc = []

   def on_epoch_end(self, batch, logs={}):
      self.train_loss.append(logs.get('loss'))
      self.train_acc.append(logs.get('acc'))
      self.val_loss.append(logs.get('val_loss'))
      self.val_acc.append(logs.get('val_acc'))
	
class EvaluateTest(Callback):
   def __init__(self, X, y, class_type):
      self.X = X
      self.y = y
      self.class_type = class_type
   def on_train_begin(self, logs=None):
      self.F1 = []
      self.mP = []
      self.mR = []
       
      self.multilabel = []
      self.zeroAcc = []
		
   def on_epoch_end(self, batch, logs={}):
      self.evalYHat = self.model.predict(x=self.X)

      accs, mP, mR, F1 = m.computeMultiLabelLoss(self.y, self.evalYHat)
      self.multilabel.append(accs)
      self.mP.append(mP)
      self.mR.append(mR)
      self.F1.append(F1)
		
			
   def on_train_end(self, logs=None):
      self.multilabel = np.array(self.multilabel)