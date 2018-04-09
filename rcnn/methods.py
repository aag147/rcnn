# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:30:30 2018

@author: aag14
"""

from models import AlexNet, PairWiseStream, getWeightsURL
from keras.layers import Add, Activation
from keras.models import Model

def _final_stop(inputs, outputs, include_weights, task):
    if task == 'multi-label':
        outputs = Activation("sigmoid",name="predictions")(outputs)
    else:
        outputs = Activation("softmax",name="predictions")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    if type(include_weights)=='str' and len(include_weights) > 0:
        model.load_weights(getWeightsURL()+include_weights)
    return model
        
def HO_RCNN_Weights():
    return "alexnet_weights.h5"

def HO_RCNN(nb_classes, include_weights=True, task='multi-label'):
    weights = HO_RCNN_Weights() if include_weights == True else False
    modelPrs = AlexNet(weights, nb_classes, include='fc')
    modelObj = AlexNet(weights, nb_classes, include='fc')
    modelPar = PairWiseStream(nb_classes = nb_classes, include='fc')             
    outputs = Add()([modelPrs.output, modelObj.output, modelPar.output])
    
    inputs = [modelPrs.input, modelObj.input, modelPar.input]
    final_model = _final_stop(inputs, outputs, include_weights, task)
    return final_model
        
def HO_RCNN_2(nb_classes, include_weights=False, task='multi-label'):
    weights = HO_RCNN_Weights() if include_weights else False
    modelPrs = AlexNet(weights, nb_classes, include='fc')
    modelObj = AlexNet(weights, nb_classes, include='fc')            
    outputs = Add()([modelPrs.output, modelObj.output])
    
    inputs = [modelPrs.input, modelObj.input]
    final_model = _final_stop(inputs, outputs, include_weights, task)
    return final_model