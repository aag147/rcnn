# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import utils
import numpy as np
import copy as cp
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


#%%
class LogHistory():
    def __init__(self):
      self.train_loss = []
      self.train_acc = []
      self.val_loss = []
      self.val_acc = []
      
    def newEpoch(self, logs):
      self.train_loss.append(logs.get('loss'))
      self.train_acc.append(logs.get('acc'))
      self.val_loss.append(logs.get('val_loss'))
      self.val_acc.append(logs.get('val_acc'))

class EvalResults():
    def __init__(self, model, gen):
      self.tp = []
      self.fp = []
      self.fn = []
      self.precision = []
      self.recall = []
      
      self.mP = None
      self.mR = None
      self.F1 = None
       
      self.nb_zeros = None
      
      self.model=model
      self.gen=gen
      self.Y_hat = None
      
      self._evaluate()

    def _evaluate(self):
      evalYHat = np.zeros([self.gen.nb_samples, self.gen.nb_classes])
      Y = np.zeros([self.gen.nb_samples, self.gen.nb_classes])
      iterGen = self.gen.begin()
      for i in range(self.gen.nb_batches):
          utils.update_progress(i / self.gen.nb_batches)
          batch, y = next(iterGen)
          y_hat = self.model.predict_on_batch(x=batch)
          s_idx = i * self.gen.batch_size
          f_idx = s_idx + len(batch[0])
          evalYHat[s_idx:f_idx, :] = y_hat
          Y[s_idx:f_idx, :] = y
      utils.update_progress(self.gen.nb_batches)
      print()
      accs, self.mP, self.mR, self.F1 = computeMultiLabelLoss(Y, evalYHat)
      self.tp = accs[:,1]
      self.fp = accs[:,2]
      self.fn = accs[:,3]
      self.precision = accs[:,4]
      self.recall = accs[:,5]
      self.nb_zeros = np.count_nonzero(accs[:,1] == 0)
      self.Y_hat = evalYHat


#%% COMPUTE ACCURACY
def computeLoss(Y, Y_hat, top):
    acc = 0.0
    for i in range(len(Y)):
        y = Y[i]
        p = Y_hat[i,:]
        I = np.argsort(p)[::-1]
        pTop = I[0:top]
        if y in pTop:
            acc = acc+1
    if len(Y)==0:
        acc = 2
    else:
        acc = acc / len(Y)
    loss = 1 - acc
    return acc
        
def computeConfusionMatrix(Y, Y_hat):
    Y = utils.getVectorLabels(Y)
    Y_hat = utils.getVectorLabels(Y_hat)
    return confusion_matrix(Y, Y_hat)

def computeMultiLabelLoss(Y, Y_hat):
    (nb_samples, nb_classes) = Y_hat.shape
    Y_hat = cp.copy(Y_hat)
    Y_hat[Y_hat>=0.5] = 1
    Y_hat[Y_hat<0.5] = 0
    accs = np.zeros((nb_classes,6))
    for x in range(nb_classes):
        y_total = 0
        tp = 0
        fp = 0
        fn = 0
        APs = []
        for i in range(nb_samples):
            y = Y[i,x]
            y_hat = Y_hat[i,x]
            if not y and y_hat:
                fp += 1
            if y and y_hat:
                tp += 1
            if y and not y_hat:
                fn += 1
            
        y_total = tp+fn
        p = tp / (tp+fp) if tp>0 else 0.0
        r = tp / (tp+fn) if tp>0 else 0.0
        
#        AP = average_precision_score(Y, Y_hat)
#        mAP = np.mean(AP)
        accs[x,:] = [y_total, tp, fp, fn, p, r]
        
    mP = np.mean(accs[:,4])
    mR = np.mean(accs[:,5])
    F1 = 0.0 if mP+mR==0 else 2 * ((mP * mR) / (mP + mR))
    return accs, mP, mR, F1

def computeIndividualLabelLoss(Y, Y_hat):
    accs = np.zeros(16)
    for x in range(16):
        y_hat_x = []
        testY_x = []
        for i in range(len(Y)):
            if Y[i]==x:
                testY_x.append(x)
                y_hat_x.append(Y_hat[i,:])
        testY_x = np.array(testY_x)
        y_hat_x = np.array(y_hat_x)
        acc1  = computeLoss(testY_x, y_hat_x, 1)
        accs[x] = acc1
#        print("[%d] TOP 1: " %  (x), acc1)
    return accs

if __name__ == "__main__":
    print("hej")
