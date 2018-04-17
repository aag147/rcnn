# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:30:30 2018

@author: aag14
"""

from models import AlexNet, PairWiseStream, classifier, input_rois
from keras.layers import Add, Activation
from keras.models import Model
import numpy as np

def _final_stop(inputs, outputs, cfg):
    if cfg.task == 'multi-label':
        outputs = Activation("sigmoid",name="predictions")(outputs)
    else:
        outputs = Activation("softmax",name="predictions")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        model.load_weights(cfg.weights_path+cfg.my_weights)
    return model
        
def HO_RCNN_Weights(cfg):
    return cfg.weights_path + "alexnet_weights.h5"

def HO_RCNN(cfg):
    weights = HO_RCNN_Weights(cfg) if cfg.pretrained_weights == True else False
    modelPrs = AlexNet(weights, cfg.nb_classes, include='fc')
    modelObj = AlexNet(weights, cfg.nb_classes, include='fc')
    modelPar = PairWiseStream(nb_classes = cfg.nb_classes, include='fc')             
    
    models = [modelPrs, modelObj, modelPar]
    models = [models[i] for i in range(len(models)) if cfg.inputs[i]]
    
    assert len(models)>0, 'minimum one model must be included in method'
    if len(models) == 1:
        outputs = models[0].output
    else:
        outputs = Add()([model.output for model in models])
    inputs = [model.input for model in models]
    
    final_model = _final_stop(inputs, outputs, cfg)
    return final_model


def Fast_HO_RCNN(cfg):
    weights = HO_RCNN_Weights(cfg) if cfg.pretrained_weights == True else False
    modelShr = AlexNet(weights, cfg.nb_classes, include='none')
    prsRoI   = input_rois()
    objRoI   = input_rois()
    modelPrs = classifier(modelShr, prsRoI, cfg.nb_classes)
    modelObj = classifier(modelShr, objRoI, cfg.nb_classes)
    modelPar = PairWiseStream(nb_classes = cfg.nb_classes, include='fc')             
    outputs = Add()([modelPrs.output, modelObj.output, modelPar.output])
    
    inputs = [modelShr.input, prsRoI, objRoI, modelPar.input]
    final_model = _final_stop(inputs, outputs, cfg)
    return final_model

