# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import numpy as np
import copy as cp
from keras import backend as K

#%% COMPUTE ACCURACY
def weigthed_binary_crossentropy(wp, wn):
    def weighted_loss(y_true, y_pred):
        return K.sum((wn*(1-y_true)+wp*(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss
