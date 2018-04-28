# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:30:30 2018

@author: aag14
"""

from models import AlexNet, VGG16, PairWiseStream, classifier, input_rois
from keras.layers import Add, Activation
from keras.models import Model
import numpy as np
import os
from keras import backend as K


def _final_stop(inputs, outputs, cfg):
    if cfg.task == 'multi-label':
        outputs = Activation("sigmoid",name="predictions")(outputs)
    else:
        outputs = Activation("softmax",name="predictions")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        print('Loading my weights...')
        path = cfg.my_weights_path + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        model.load_weights(path)         
    return model
        
def AlexNet_Weights(cfg):
    return cfg.weights_path + "alexnet_weights.h5"
def VVG16_Weights(cfg):
    return cfg.weights_path + "vgg16_weights_tf.h5"

def HO_RCNN(cfg):
    K.set_image_dim_ordering('th')
    weights = AlexNet_Weights(cfg) if cfg.pretrained_weights == True else False
    modelPrs = AlexNet((3, 227, 227), weights, cfg.nb_classes, include='fc')
    modelObj = AlexNet((3, 227, 227), weights, cfg.nb_classes, include='fc')
    modelPar = PairWiseStream(input_shape=(2,64,64), nb_classes = cfg.nb_classes, include='fc')             
    
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
    K.set_image_dim_ordering('tf')
    weights = VVG16_Weights(cfg) if cfg.pretrained_weights == True else False
#    modelShr = AlexNet(weights, cfg.nb_classes, include='none')
    modelShr = VGG16((cfg.ydim, cfg.xdim, cfg.cdim), weights, cfg.nb_classes, include='none')
    prsRoI   = input_rois()
    objRoI   = input_rois()
    print(modelShr.layers[-1].output_shape)
    modelPrs = classifier(modelShr.output, prsRoI, cfg, nb_classes=cfg.nb_classes)
    modelObj = classifier(modelShr.output, objRoI, cfg, nb_classes=cfg.nb_classes)
    modelPar = PairWiseStream(input_shape=(64,64,2), nb_classes = cfg.nb_classes, include='fc')      
    
    outputs = [modelPrs, modelObj, modelPar.output]
    outputs = [outputs[i] for i in range(len(outputs)) if cfg.inputs[i]]
    
    if sum(cfg.inputs) == 1:
        outputs = outputs[0]
    else:
        
        outputs = Add()(outputs)
        
    inputs = [prsRoI, objRoI, modelPar.input]
    inputs = [inputs[i] for i in range(len(inputs)) if cfg.inputs[i]]
    
    if cfg.inputs[0] or cfg.inputs[1]:
        inputs = [modelShr.input] + inputs    
    
    final_model = _final_stop(inputs, outputs, cfg)
    
    return final_model

def Pretrained_HO_RCNN(cfg):
    K.set_image_dim_ordering('th')
    weights =  False
    modelPrs = AlexNet((3, 227, 227), weights, cfg.nb_classes, include='fc')
    modelObj = AlexNet((3, 227, 227), weights, cfg.nb_classes, include='fc')
    modelPar = PairWiseStream(input_shape=(2,64,64), nb_classes = cfg.nb_classes, include='fc')             
    
    my_actual_weights_path = cfg.my_weights_path
    
    cfg.my_weights_path = cfg.prs_weights_path
    cfg.my_weights     = cfg.prs_weights
    modelPrs = _final_stop(modelPrs.input, modelPrs.output, cfg)
    cfg.my_weights_path = cfg.obj_weights_path
    cfg.my_weights     = cfg.obj_weights
    modelObj = _final_stop(modelObj.input, modelObj.output, cfg)
    cfg.my_weights_path = cfg.par_weights_path
    cfg.my_weights     = cfg.par_weights
    modelPar = _final_stop(modelPar.input, modelPar.output, cfg)
    
    cfg.my_weights_path = my_actual_weights_path
    cfg.my_weights = None
    
    models = [modelPrs, modelObj, modelPar]
    outputs = Add()([model.output for model in models])
    inputs = [model.input for model in models]
    
    final_model = _final_stop(inputs, outputs, cfg)
    return final_model