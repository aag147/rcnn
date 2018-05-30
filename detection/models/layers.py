# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:16:45 2018

@author: aag14
"""

from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, TimeDistributed

import tensorflow as tf

import numpy as np


def rpn(x):
    base_layers = x[0]
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.01), name='rpn_conv1')(
        base_layers)

    return x

def fullyConnected(x):
    assert(len(x) == 1)
    rois = x[0]
    
    dense_1 = TimeDistributed(Flatten())(rois)
    dense_1 = TimeDistributed(
        Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))
    )(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = TimeDistributed(
        Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))
    )(dense_1)
    dense_2 = Dropout(0.5)(dense_2)

    return dense_2


class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    Source: https://github.com/jinfagang/keras_frcnn/blob/master/keras_frcnn/roi_pooling_conv.py
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (ymin,xmin,ymax,xmax)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, **kwargs):
        assert K.image_dim_ordering() in {'tf'}, 'dim_ordering must be in {tf}'
        self.pool_size = pool_size
        self.nb_channels = None

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]
        super(RoiPoolingConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return 1, None, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]
        rois = K.squeeze(rois, axis=0)

        box_ind = K.cast(rois[:,0], 'int32')
        rois    = K.cast(rois[:,1:], 'float')
#        rois = tf.stop_gradient(rois)
#        box_ind = tf.stop_gradient(box_ind)
        final_output = tf.image.crop_and_resize(img, boxes=rois, box_ind=box_ind, crop_size=(self.pool_size, self.pool_size), method="bilinear")
#        final_output = tf.stack(final_output) 
        final_output = K.expand_dims(final_output, axis=0)
        return final_output
    
    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "nb_channels": self.nb_channels
        }
        base_config = super(RoiPoolingConv, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))