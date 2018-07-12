# -*- coding: utf-8 -*-
"""
Created on Tue May 22 20:08:04 2018

@author: aag14
"""
import keras
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras import backend as K

def VGG16_buildin(cfg):
    print('   Using built in KERAS VGG16 model')
    def _vgg(input_image):
        model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_image)
        for i, layer in enumerate(model.layers):
            layer.kernel_regularizer = l2(cfg.weight_decay_shared)
            layer.bias_regularizer   = l2(cfg.weight_decay_shared)
        
        return model.layers[-2].output
    return _vgg

def AlexNet_buildin(cfg):
    print('   Using own AlexNet model')
    def _alex(input_image):
        model = AlexNet(include_top = not cfg.do_fast_hoi, weights='imagenet', input_tensor=input_image, cfg=cfg)
        for i, layer in enumerate(model.layers):
            layer.kernel_regularizer = l2(cfg.weight_decay)
            layer.bias_regularizer   = l2(cfg.weight_decay)
        
        return model.layers[-2].output
    return _alex
    
def AlexNet(include_top = False, weights=None, input_tensor=None, cfg=None):
    #https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/convnets-keras/convnetskeras/convnets.py
    inputs = input_tensor
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

    
    if weights is not None and not include_top:
        weights_path = cfg.weights_path + "alexnet_weights_tf_notop.h5"
        model.load_weights(weights_path)
    
    
    if include_top:
        dense_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_5)
        dense_1 = Flatten()(dense_1)
        dense_1 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_1)
        dense_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_1)
        dense_2 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, kernel_initializer=RandomNormal(stddev=0.01), kernel_regularizer= l2(cfg.weight_decay), bias_regularizer = l2(cfg.weight_decay))(dense_2)
        model = Model(inputs=inputs, outputs=dense_3)
        
        if weights is not None:
            weights_path = cfg.weights_path + "alexnet_weights_tf.h5"
            model.load_weights(weights_path)
        
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


def VGG16(cfg):
    print('   Using own VGG16 model')
    #https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    #https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
    def _vgg(input_image):
        # First component #
        x = Conv2D(
                64, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv1a'
            )(input_image)
        x = Conv2D(
                64, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv1b'
            )(x)
        x = MaxPooling2D(
                (2,2), strides=(2,2),
                name='max1'
            )(x)
        
        # Second component #
        x = Conv2D(
                128, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv2a'
            )(x)
        x = Conv2D(
                128, (3, 3), 
                activation='relu', padding='same', 
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv2b'
            )(x)
        x = MaxPooling2D(
                (2,2), strides=(2,2),
                name='max2'
            )(x)
    
        # Third component #
        x = Conv2D(
                256, (3, 3), 
                activation='relu', padding='same', 
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv3a'
            )(x)
        x = Conv2D(
                256, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv3b'
            )(x)
        x = Conv2D(
                256, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv3c'
            )(x)
        x = MaxPooling2D(
                (2,2), strides=(2,2),
                name='max3'
            )(x)
    
        # Fourth component #
        x = Conv2D(
                512, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv4a'
            )(x)
        x = Conv2D(
                512, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv4b'
            )(x)
        x = Conv2D(
                512, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv4c'
            )(x)
        x = MaxPooling2D(
                (2,2), strides=(2,2),
                name='max4'
            )(x)
        
        # Fifth component #
        x = Conv2D(
                512, (3, 3), 
                activation='relu', padding='same', 
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv5a'
            )(x)
        x = Conv2D(
                512, (3, 3), 
                activation='relu', padding='same', 
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv5b'
            )(x)
        x = Conv2D(
                512, (3, 3),
                activation='relu', padding='same',
                kernel_regularizer = l2(cfg.weight_decay),
                bias_regularizer   = l2(cfg.weight_decay),
                name='conv5c'
            )(x)
        
        # Done #    
        model = Model(inputs=input_image, outputs=x)
    
        if cfg.weights_path is not None:
            model.load_weights(cfg.weights_path + 'vgg16_weights_tf_notop.h5')
        return model.output


    return _vgg
