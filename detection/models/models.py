# -*- coding: utf-8 -*-
"""
Created on Tue May 22 20:08:04 2018

@author: aag14
"""

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2

def VGG16(cfg):
    #https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    #https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
    def _vgg(input_image):
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1a', kernel_regularizer = l2(cfg.weight_decay))(input_image)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1b', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = MaxPooling2D((2,2), strides=(2,2), name='max1')(x)
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2a', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2b', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = MaxPooling2D((2,2), strides=(2,2), name='max2')(x)
    
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3a', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3b', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3c', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = MaxPooling2D((2,2), strides=(2,2), name='max3')(x)
    
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4a', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4b', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4c', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = MaxPooling2D((2,2), strides=(2,2), name='max4')(x)
    
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5a', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5b', kernel_regularizer = l2(cfg.weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5c', kernel_regularizer = l2(cfg.weight_decay))(x)
    
        model = Model(inputs=input_image, outputs=x)
    
        if cfg.weights_path is not None:
            model.load_weights(cfg.weights_path + 'vgg16_weights_tf_notop.h5')
        return model.output


    return _vgg
