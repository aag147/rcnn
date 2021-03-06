# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:30:30 2018

@author: aag14
"""

from models import AlexNet, VGG16, PairWiseStream, classifier, input_rois, AlexNet_tf, fastClassifier, fastPairWiseStream, RoiPoolingConv
from keras.layers import Add, Activation
from keras.models import Model
import numpy as np
import os
from keras import backend as K
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers

import losses

def _final_stop(inputs, outputs, cfg):
    if cfg.task == 'multi-label':
        outputs = Activation("sigmoid",name="predictions")(outputs)
    else:
        outputs = Activation("softmax",name="predictions")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    if type(cfg.my_weights)==str and len(cfg.my_weights) > 0:
        print('Loading my weights...')        
        loss_ce  = losses.weigthed_binary_crossentropy(cfg.wp,1)
        roi_pooling = RoiPoolingConv
        get_custom_objects().update({"weighted_loss": loss_ce})
        get_custom_objects().update({'RoiPoolingConv': roi_pooling})
        
        path = cfg.my_weights_path + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        if cfg.only_use_weights:
            model.load_weights(path)
        else:
            model = load_model(path)
        
#    for layer in model.layers:
#        if hasattr(layer, 'kernel_regularizer'):
#            layer.kernel_regularizer= regularizers.l2(cfg.weight_decay+100000)
    return model
        
def AlexNet_Weights_th(cfg):
    return cfg.weights_path + "alexnet_weights.h5"


def AlexNet_Weights(cfg):
    return cfg.weights_path + "alexnet_weights_tf.h5"
def AlexNet_Weights_notop(cfg):
    return cfg.weights_path + "alexnet_weights_tf_notop.h5"
def VGG16_Weights(cfg):
    return cfg.weights_path + "vgg16_weights_tf.h5"
def VGG16_Weights_notop(cfg):
    return cfg.weights_path + "vgg16_weights_tf_notop.h5"

def HO_RCNN(cfg):
    K.set_image_dim_ordering('th')
    weights = AlexNet_Weights_th(cfg) if cfg.pretrained_weights == True else False
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

def HO_RCNN_tf(cfg):
    K.set_image_dim_ordering('tf')
    weights = AlexNet_Weights(cfg) if cfg.pretrained_weights == True else False
    modelPrs = AlexNet_tf((227, 227, 3), weights, cfg.nb_classes, cfg=cfg, include='fc')
    modelObj = AlexNet_tf((227, 227, 3), weights, cfg.nb_classes, cfg=cfg, include='fc')
    modelPar = PairWiseStream(input_shape=(64,64,2), nb_classes = cfg.nb_classes, cfg=cfg, include='fc')             
    
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
    if cfg.backbone == 'vgg':
        print('Use VGG16 backbone')
        weights = VGG16_Weights_notop(cfg) if cfg.pretrained_weights == True else False
        modelShr = VGG16((None, None, cfg.cdim), weights, cfg.nb_classes, include='basic')
    else:
        print('Use Alexnet backbone')
        weights = AlexNet_Weights_notop(cfg) if cfg.pretrained_weights == True else False
        modelShr = AlexNet_tf((None, None, cfg.cdim), weights, cfg.nb_classes, cfg=cfg, include='basic')
    prsRoI   = input_rois()
    objRoI   = input_rois()
    modelPrs = fastClassifier(modelShr.output, prsRoI, cfg, nb_classes=cfg.nb_classes)
    modelObj = fastClassifier(modelShr.output, objRoI, cfg, nb_classes=cfg.nb_classes)
    modelPar = fastPairWiseStream(input_shape=(None,64,64,2), cfg=cfg, nb_classes = cfg.nb_classes, include='fc')      
    
    if cfg.backbone == 'vgg':
        # Only train from conv3_1
        for i, layer in enumerate(modelShr.layers):
            layer.trainable = False
            if i > 6:
                break
    
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
    outputs = [model.layers[-2].output for model in models]
    outputs = Add()(outputs)
    inputs = [model.input for model in models]
    
    final_model = _final_stop(inputs, outputs, cfg)
    return final_model
