# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import numpy as np
import plotData as pd
import copy as cp
from sklearn.metrics import average_precision_score

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
    

def computeMultiLabelLoss(Y, Y_hat):
    (nb_samples, nb_classes) = Y_hat.shape
    Y_hat = cp.copy(Y_hat)
    Y_hat[Y_hat>=0.5] = 1
    Y_hat[Y_hat<0.5] = 0
    accs = np.zeros((16,8))
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
        
        AP = average_precision_score(Y, Y_hat)
        mAP = np.mean(AP)
        accs[x,:] = [y_total, tp, fp, fn, p, r]
        
    mP = np.mean(accs[:,4])
    mR = np.mean(accs[:,5])
    F1 = 2 * ((mP * mR) / (mP + mR))
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
