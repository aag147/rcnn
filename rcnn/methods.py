# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:30:30 2018

@author: aag14
"""

from models import AlexNet, PairWiseStream
from keras.layers import Add, Activation
from keras.models import Model
import numpy as np

def _final_stop(inputs, outputs, my_weights, cfg):
    if cfg.task == 'multi-label':
        outputs = Activation("sigmoid",name="predictions")(outputs)
    else:
        outputs = Activation("softmax",name="predictions")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    if type(my_weights)==str and len(my_weights) > 0:
        model.load_weights(cfg.weights_path+my_weights)
    return model
        
def HO_RCNN_Weights(cfg):
    return cfg.weights_path + "alexnet_weights.h5"

def HO_RCNN(cfg, my_weights=None):
    weights = HO_RCNN_Weights(cfg) if cfg.pretrained_weights == True else False
    modelPrs = AlexNet(weights, cfg.nb_classes, include='fc')
    modelObj = AlexNet(weights, cfg.nb_classes, include='fc')
    modelPar = PairWiseStream(nb_classes = cfg.nb_classes, include='fc')             
    outputs = Add()([modelPrs.output, modelObj.output, modelPar.output])
    
    inputs = [modelPrs.input, modelObj.input, modelPar.input]
    final_model = _final_stop(inputs, outputs, my_weights, cfg)
    return final_model
        
def HO_RCNN_2(cfg, my_weights=None):
    weights = HO_RCNN_Weights(cfg) if cfg.pretrained_weights else False
    modelPrs = AlexNet(weights, cfg.nb_classes, include='fc')
    modelObj = AlexNet(weights, cfg.nb_classes, include='fc')            
    outputs = Add()([modelPrs.output, modelObj.output])
    
    inputs = [modelPrs.input, modelObj.input]
    final_model = _final_stop(inputs, outputs, my_weights, cfg)
    return final_model
