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

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.layer_utils import convert_all_kernels_in_model


def getWeightsURL():
    urlWeights = 'C:/Users/aag14/Documents/Skole/Speciale/Weights/'
    return urlWeights

def replaceLastLayer(model, nb_classes):    
    # Build final model
    output = Dense(nb_classes)(model.layers[-2].output)
    model = Model(input=model.input, output=output)
    return model

def VGG_16(weights_path=None, nb_classes=1000):
    #https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    #https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py
    input_shape=(3,224,224)
    model = Sequential(name='VGG16')
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'), input_shape=input_shape)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu'), padding='same')
    model.add(Conv2D(256, (3, 3), activation='relu'), padding='same')
    model.add(Conv2D(256, (3, 3), activation='relu'), padding='same')
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu'), padding='same')
    model.add(Conv2D(512, (3, 3), activation='relu'), padding='same')
    model.add(Conv2D(512, (3, 3), activation='relu'), padding='same')
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, (3, 3), activation='relu'), padding='same')
    model.add(Convolution2D(512, (3, 3), activation='relu'), padding='same')
    model.add(Convolution2D(512, (3, 3), activation='relu'), padding='same')
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def PairWiseStream(weights_path=None, nb_classes=1000, include = 'all'):
    inputs = Input(shape=(2,64,64))
    model = Sequential()
    conv_1 = Conv2D(64, (5, 5), activation='relu')(inputs)
    conv_1 = MaxPooling2D((2,2), strides=(2,2))(conv_1)
    
    conv_2 = Conv2D(32, (5, 5), activation='relu')(conv_1)
    conv_2 = MaxPooling2D((2,2), strides=(2,2))(conv_2)
    
    fc = Flatten()(conv_2)
    fc = Dense(256, activation='relu')(fc)
    fc = Dense(1000, activation='relu')(fc)
    
    model = Model(input=inputs, output=fc)
    
    if weights_path:
        model.load_weights(weights_path)
       
    if nb_classes:
        model = replaceLastLayer(model, nb_classes)
    
    if include == 'all':
        prediction = Activation("sigmoid")(model.layers[-1].output)
        model = Model(input=model.input, output=prediction)
    return model



def AlexNet(weights_path=None, nb_classes=1000, include = 'all'):
    #https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/convnets-keras/convnetskeras/convnets.py
    inputs = Input(shape=(3,227,227))

    conv_1 = Conv2D(96, (11, 11),subsample=(4,4),activation='relu')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization()(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu")(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu")(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1)

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu")(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1)

    
    dense_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_5)
    dense_1 = Flatten()(dense_1)
    dense_1 = Dense(4096, activation='relu')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000)(dense_3)
    model = Model(input=inputs, output=dense_3)

    if weights_path:
        model.load_weights(weights_path)
        
    if K.backend() == 'tensorflow':
       convert_all_kernels_in_model(model)
       
    
    if include not in {'fc', 'all'}:
        output = model.layers[-8].output
        model = Model(input=model.input, output=output)
    else:
        if nb_classes:
            model = replaceLastLayer(model, nb_classes)
    
        if include == 'all':
            prediction = Activation("sigmoid")(model.layers[-1].output)
            model = Model(input=model.input, output=prediction)

    return model


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
