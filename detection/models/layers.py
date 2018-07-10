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
from keras.regularizers import l2

import tensorflow as tf

import numpy as np


def rpn(cfg):
    def rpnFixed(x):
        base_layers = x[0]
        x = Conv2D(
                512, (3, 3),
                padding='same', activation='relu', 
                kernel_initializer = RandomNormal(stddev=0.01), 
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='rpn_conv1'
            )(base_layers)
        return x
    return rpnFixed

def slow_pooling(cfg):
    def _slow(x):
        base_layers = x[0]
        x = MaxPooling2D(
                (3, 3), 
                strides=(2,2)
            )(base_layers)
        x = K.expand_dims(
                x=x, 
                axis=0
            )
        return x
    return _slow

def slow_expansion(cfg):
    def lb_func(x):
        base_layers = x[0]
        y = K.reshape(
                x=base_layers, 
                shape=(1,-1,4096)
            )
        return y
    
    def _slow(x):
        y = Lambda(lb_func, output_shape=(None,4096))(x)
        return y
    
    return _slow

def intct_expansion(cfg):
    def lb_func(x):
        base_layers = x[0]
        y = K.reshape(
                x=base_layers, 
                shape=(1,-1,64,64,2)
            )
        return y
    
    def _slow(x):
        y = Lambda(lb_func, output_shape=(None,64,64,2))(x)
        return y
    
    return _slow

def intct_reduction(cfg):
    def lb_func(x):
        base_layers = x[0]
        y = K.reshape(
                x=base_layers, 
                shape=(-1, cfg.nb_hoi_classes)
            )
        return y
    
    def _slow(x):
        y = Lambda(lb_func, output_shape=(cfg.nb_hoi_classes,))(x)
        return y
    
    return _slow

def fullyConnected(cfg, stream=None, use_dropout=True):
    def fullyConnectedFixed(x):
        assert(len(x) == 1)
        rois = x[0]
        
        dense_1 = TimeDistributed(
            Flatten(),
            name = '%s_flatten' % stream
        )(rois)
                
        dense_1 = TimeDistributed(
            Dense(
                4096,
                activation='relu',
                kernel_initializer = RandomNormal(stddev=0.01),
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay)
            ),
            name = '%s_fc1' % stream
        )(dense_1)
        if use_dropout:
            dense_1 = Dropout(
                rate=0.5,
                name = '%s_dropout1' % stream
            )(dense_1)
        
        dense_2 = TimeDistributed(
            Dense(
                4096,
                activation='relu',
                kernel_initializer = RandomNormal(stddev=0.01),
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay)
            ),
            name = '%s_fc2' % stream
        )(dense_1)
        
        if use_dropout:
            dense_2 = Dropout(
                rate=0.5,
                name = '%s_dropout2' % stream
            )(dense_2)
    
        return dense_2
    
    return fullyConnectedFixed


def pairwiseStream(cfg):
    def pairwiseStreamFixed(x):
        pattern = x[0]
        conv_1 = TimeDistributed(
            Conv2D(
                64, (5, 5),
                activation='relu',
                kernel_initializer=RandomNormal(stddev=0.01),
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay)
            ),
            name = 'pairwise_conv1a'
        )(pattern)
        
        conv_1 = TimeDistributed(
            MaxPooling2D((2,2), strides=(2,2)),
            name = 'pairwise_max1'
        )(conv_1)
    
        conv_2 = TimeDistributed(
            Conv2D(
                32, (5, 5),
                activation='relu',
                kernel_initializer=RandomNormal(stddev=0.01),
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay)
            ),
            name = 'pairwise_conv2a'
        )(conv_1)
        
        conv_2 = TimeDistributed(
            MaxPooling2D((2,2), strides=(2,2)),
            name = 'pairwise_max2'
        )(conv_2)
        
        dense1 = TimeDistributed(
            Flatten(),
            name = 'pairwise_flatten'
        )(conv_2)
        
        dense1 = TimeDistributed(
            Dense(
                256,
                activation='relu',
                kernel_initializer=RandomNormal(stddev=0.01),
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay)
            ),
            name = 'pairwise_fc1'
        )(dense1)
        
        return dense1
    
    return pairwiseStreamFixed


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
    def __init__(self, pool_size=3, nb_channels=3, **kwargs):
        super(RoiPoolingConv, self).__init__(**kwargs)
#        assert K.image_dim_ordering() in {'tf'}, 'dim_ordering must be in {tf}'
        self.pool_size = pool_size
        self.nb_channels = 3

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
            "pool_size": self.pool_size
        }
        base_config = super(RoiPoolingConv, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))