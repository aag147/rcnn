# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:07:29 2018

@author: aag14
"""
import keras
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer

from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, TimeDistributed

import tensorflow as tf

import numpy as np


class InteractionOverUnion(Layer):
    def __init__(self, **kwargs):

        super(InteractionOverUnion, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][1]
        super(InteractionOverUnion, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return 1, 2, None, self.nb_channels
    
    
    def iou(self, outputs, targets):
        target_area = (targets[:, 2] - targets[:, 0] + 1) * (targets[:, 3] - targets[:, 1] + 1)
        output_area = keras.backend.expand_dims((outputs[:, 2] - outputs[:, 0] + 1) * (outputs[:, 3] - outputs[:, 1] + 1), 1) 
        
        intersection_c_minimum = keras.backend.minimum(keras.backend.expand_dims(outputs[:, 2], 1), targets[:, 2])
        intersection_c_maximum = keras.backend.maximum(keras.backend.expand_dims(outputs[:, 0], 1), targets[:, 0])
    
        intersection_r_minimum = keras.backend.minimum(keras.backend.expand_dims(outputs[:, 3], 1), targets[:, 3])
        intersection_r_maximum = keras.backend.maximum(keras.backend.expand_dims(outputs[:, 1], 1), targets[:, 1])
    
        intersection_c = intersection_c_minimum - intersection_c_maximum + 1
        intersection_r = intersection_r_minimum - intersection_r_maximum + 1
    
        intersection_c = keras.backend.maximum(intersection_c, 0)
        intersection_r = keras.backend.maximum(intersection_r, 0)
        intersection_area = intersection_c * intersection_r
    
        union_area = output_area + target_area - intersection_area
    
        union_area = keras.backend.maximum(union_area, keras.backend.epsilon())

        iou = intersection_area / union_area
        return iou

    def call(self, x):

        assert(len(x) == 4)
        
        human_outputs = x[0][0]
        object_outputs = x[1][0]
        human_targets = x[2][0]
        object_targets = x[3][0]

        print(human_outputs.shape)

        human_ious = self.iou(human_outputs, human_targets)
        object_ious = self.iou(object_outputs, object_targets)
        human_ious = keras.backend.expand_dims(human_ious, axis=0)
        object_ious = keras.backend.expand_dims(object_ious, axis=0)
        
        ious = keras.backend.concatenate([human_ious, object_ious], axis=0)
        ious = keras.backend.expand_dims(ious, 0) 
        print(ious.shape)
        return  ious
    
    def get_config(self):
        configuration = {
            "nb_channels": self.nb_channels
        }
        return {**super(InteractionOverUnion, self).get_config(), **configuration}

