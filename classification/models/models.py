# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:37:03 2018

@author: aag14
"""

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.initializers import TruncatedNormal, RandomNormal, Ones
import keras.applications as keras_models
from keras import regularizers
from keras.regularizers import l2

from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.layer_utils import convert_all_kernels_in_model

import tensorflow as tf


def VGG16(input_shape, weights_path=None, nb_classes=1000, include='all'):
    #https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    #https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
    model = Sequential(name='VGG16')
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    
    if include == 'fc':
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000))

    model = final_model(model, weights_path, nb_classes, include)

    return model


def AlexNet(input_shape, weights_path=None, nb_classes=1000, cfg=None, include = 'all'):
    #https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/convnets-keras/convnetskeras/convnets.py
    inputs = Input(shape=input_shape)
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization()(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = concatenate([
        Conv2D(128, (5,5), activation="relu", kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], axis=1)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Conv2D(384, (3,3), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = concatenate([
        Conv2D(192, (3,3), activation="relu", kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], axis=1)

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = concatenate([
        Conv2D(128, (3,3), activation="relu", kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], axis=1)

    
    dense_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_5)
    dense_1 = Flatten()(dense_1)
    dense_1 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay))(dense_3)
    model = Model(inputs=inputs, outputs=dense_3)

    model = final_model(model, weights_path, nb_classes, include)

    return model

def PairWiseStream(input_shape, weights_path=None, nb_classes=1000, cfg=None, include = 'all'):
    inputs = Input(shape=input_shape)
    model = Sequential()
    conv_1 = Conv2D(64, (5, 5), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(inputs)
    conv_1 = MaxPooling2D((2,2), strides=(2,2))(conv_1)
    
    conv_2 = Conv2D(32, (5, 5), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(conv_1)
    conv_2 = MaxPooling2D((2,2), strides=(2,2))(conv_2)
    
    fc = Flatten()(conv_2)
    fc = Dense(256, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(fc)
    fc = Dense(nb_classes, kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(fc)
    
    model = Model(inputs=inputs, outputs=fc)
    
    model = final_model(model, weights_path, nb_classes, include)
    return model

def final_model(model, weights_path, nb_classes, include):    
    # Build final model
    if weights_path:
        model.load_weights(weights_path)
        
    if K.image_dim_ordering() == 'th':
       convert_all_kernels_in_model(model)
    
    if include == 'none':
        output = model.layers[-8].output
        model = Model(inputs=model.input, outputs=output)
    elif include == 'fc':
        output = Dense(nb_classes)(model.layers[-2].output)
        model = Model(inputs=model.input, outputs=output)
    return model

def input_rois():
    input_rois = Input(shape=(None, 5))
    return input_rois

def rpn(base_layers, num_anchors, cfg):
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', kernel_regularizer= l2(cfg.weight_decay), name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', kernel_regularizer= l2(cfg.weight_decay), name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', kernel_regularizer= l2(cfg.weight_decay), name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, cfg, nb_classes=21):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    cfg.pooling_regions = 7
#    print(K.shape(base_layers))
#    print(K.shape(input_rois))
    out_roi_pool = RoiPoolingConv(cfg)([base_layers, input_rois])

    dense_1 = Flatten()(out_roi_pool)
    dense_1 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_2)
    dense_3 = Dropout(0.5)(dense_2)

    out_class = Dense(nb_classes, kernel_initializer='uniform', kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_3)
    # note: no regression target for bg class
#    out_regr = Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero')(dense_3)

    return out_class



###############
##### FAST ####
###############
def fastPairWiseStream(input_shape, weights_path=None, cfg=None, nb_classes=1000, include = 'all'):
    inputs = Input(shape=input_shape)
    model = Sequential()
    conv_1 = TimeDistributed(Conv2D(64, (5, 5), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay)))(inputs)
    conv_1 = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)))(conv_1)
    
    conv_2 = TimeDistributed(Conv2D(32, (5, 5), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay)))(conv_1)
    conv_2 = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)))(conv_2)
    
    fc = TimeDistributed(Flatten())(conv_2)
    fc = TimeDistributed(Dense(256, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay)))(fc)
    fc = TimeDistributed(Dense(nb_classes, kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay)))(fc)
    
    model = Model(inputs=inputs, outputs=fc)
    
    model = final_model(model, weights_path, nb_classes, include)
    return model

def fastClassifier(base_layers, input_rois, cfg, nb_classes=21):    
    cfg.pooling_regions = 7
    out_roi_pool = RoiPoolingConv(cfg)([base_layers, input_rois])
    
    dense_1 = TimeDistributed(Flatten())(out_roi_pool)
    dense_1 = TimeDistributed(
        Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))
    )(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = TimeDistributed(
        Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))
    )(dense_1)
    dense_2 = Dropout(0.5)(dense_2)
    
    out_class = TimeDistributed(Dense(nb_classes, kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay)))(dense_2)

    return out_class

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
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, cfg, **kwargs):
        assert K.image_dim_ordering() in {'tf'}, 'dim_ordering must be in {tf}'
        self.pool_size = cfg.pool_size
#        self.num_rois = cfg.num_rois
        self.batch_size = cfg.train_cfg.batch_size

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return 1, None, self.nb_channels, self.pool_size, self.pool_size

    def call(self, x):

        assert(len(x) == 2)
        img = x[0]
        rois = x[1][0]
        box_ind = K.cast(rois[:,0], 'int32')
        rois    = rois[:,1:]

        final_output = tf.image.crop_and_resize(img, boxes=rois, box_ind=box_ind, crop_size=(self.pool_size, self.pool_size), method="bilinear")
        final_output = K.expand_dims(final_output, axis=0)
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'batch_size': self.batch_size}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        K.set_image_dim_ordering('th')
        if K.backend()=='tensorflow':
            b, ch, r, c = X.get_shape()
        else:
            b, ch, r, c = X.shape            
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , ((0,0),(half,half)))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            if K.backend()=='tensorflow':
                ch = int(ch)
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        if K.backend()=='tensorflow':
            div = int(X.get_shape()[axis]) // ratio_split
        else:
            div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output
    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)


def AlexNet_tf(input_shape, weights_path=None, nb_classes=1000, cfg=None, include = 'all'):
    #https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/convnets-keras/convnetskeras/convnets.py
    inputs = Input(shape=input_shape)
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization_tf()(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = concatenate([
        Conv2D(128, (5,5), activation="relu", kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(
            splittensor_tf(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], axis=3)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization_tf()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Conv2D(384, (3,3), activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = concatenate([
        Conv2D(192, (3,3), activation="relu", kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(
            splittensor_tf(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], axis=3)

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = concatenate([
        Conv2D(128, (3,3), activation="relu", kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(
            splittensor_tf(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], axis=3)

    
    model = Model(inputs=inputs, outputs=conv_5)
    
    if include == 'fc':
        dense_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_5)
        dense_1 = Flatten()(dense_1)
        dense_1 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_1)
#        dense_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_1)
#        dense_2 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_2)
        model = Model(inputs=inputs, outputs=dense_3)

    model = final_model(model, weights_path, nb_classes, include)
    return model

def crosschannelnormalization_tf(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
#        K.set_image_dim_ordering('th')
        if K.backend()=='tensorflow':
            b, r, c, ch = X.get_shape()
        else:
            b, r, c, ch = X.shape            
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,1,3,2))
                                              , ((0,0),(half,half)))
        extra_channels = K.permute_dimensions(extra_channels, (0,1,3,2))
        scale = k
        for i in range(n):
            if K.backend()=='tensorflow':
                ch = int(ch)
            scale += alpha * extra_channels[:,:,:,i:i+ch]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor_tf(axis=3, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        if K.backend()=='tensorflow':
            div = int(X.get_shape()[axis]) // ratio_split
        else:
            div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output
    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)