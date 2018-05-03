# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:13:36 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../layers/')
sys.path.append('../cfgs/')

from layers.backend import InteractionOverUnion
from model_trainer import model_trainer
from load_data import data
from fast_generators import DataGenerator
from methods import Fast_HO_RCNN

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, Lambda

import tensorflow as tf  
from models import RoiPoolingConv
import keras
import keras.backend as K



### Test union over intersection ###
boxes1 = np.array([[0,0,4,4], [2,2,3,3], [8,8,9,10]])
boxes2 = np.array([[0,0,3,3], [0,0,4,4], [8,8,10,10]])
boxes3 = np.array([[0,0,4,4], [0,0,4,4], [8,8,10,10]])
boxes1 = np.expand_dims(boxes1, axis=0)
boxes2 = np.expand_dims(boxes2, axis=0)
boxes3 = np.expand_dims(boxes3, axis=0)

labels = np.array([[0,0,1],[0,1,0],[1,0,0]])
labels = np.expand_dims(labels, axis=0)

human_outputs = Input(shape=(None,4))
human_targets = Input(shape=(None,4))
object_outputs = Input(shape=(None,4))
object_targets = Input(shape=(None,4))
labels_input = Input(shape=(None,3))
#    final_output = tf.image.crop_and_resize(image_input, boxes=boxes_input, box_ind=idxs_input, crop_size=(pool_size, pool_size))    
result = InteractionOverUnion()([human_outputs, object_outputs, human_targets, object_targets])


def foreground_labels(labels, x):
#    print('fg',x)
#    print('fg',labels)
    x = tf.where(x)
#    print('fg', x.shape)
    x = tf.gather_nd(labels, x)
#    print('fg', x.shape)
    x = tf.reduce_sum(x, axis=0)
    x = tf.clip_by_value(x, 0, 1)
    x = tf.cast(x, 'int32')
    return x

def foreground_boxes(human_targets, object_targets, x):
#    print('bg',x)
#    print('bg',human_targets)
#    print('bg',object_targets)
    x = tf.where(x)
    gt_humans = tf.gather_nd(human_targets, x)
    gt_objects = tf.gather_nd(object_targets, x)

    gt_humans = keras.backend.expand_dims(gt_humans, axis=0)
    gt_objects = keras.backend.expand_dims(gt_objects, axis=0)    
    boxes = keras.backend.concatenate([gt_humans, gt_objects], axis=0)
    print('boxes', boxes.shape)
    return boxes


def regression_box(fg_assignment, fg_boxes):    
    bbox_targets = tf.Variable(initial_value=np.zeros([4*3,]).tolist())

    label_idxs = tf.where(fg_assignment[0])
    label_idxs = tf.gather_nd(labels, label_idxs)
    label_idxs = tf.argmax(label_idxs, axis=1)

    start = 4 * label_idxs

    ii = K.expand_dims(tf.Variable(np.zeros([2]).tolist()),0)
    ii = K.tile(ii, [4, 1])
    ii = K.reshape(ii, (-1,))
    ii = K.cast(ii, dtype='int64')

    aa = K.expand_dims(K.concatenate([start, start + 1, start + 2, start + 3], 0), 0)
    aa = K.reshape(aa, (-1,))
    aa = K.cast(aa, dtype='int64')
    
    print('ii', ii.shape)
    print('aa', aa.shape)

#    indices = K.concatenate([ii, aa], 1)
    indices = aa
    
    updates = fg_boxes
    updates = K.transpose(updates)
    updates = K.reshape(updates, (-1,))
    
    updates = K.cast(updates, keras.backend.floatx())      

    bbox_targets = tf.scatter_add(bbox_targets, indices, updates)    
    bbox_targets = K.reshape(bbox_targets, (-1, 4))
    return bbox_targets

def regression_boxes(human_targets, object_targets, fg_assignment):
    fg_boxes = foreground_boxes(human_targets, object_targets, fg_assignment)
    human_reg = regression_box(fg_assignment, fg_boxes[0])
    object_reg = regression_box(fg_assignment, fg_boxes[1])
    return human_reg, object_reg
    

def lambda_func(x):
    # Inputs
    human_targets = x[1][0]
    object_targets = x[2][0]
    labels = x[3][0]
    x = x[0][0]
    y = 0.5
    # Finding corresponding ground truth boxes
    out = tf.greater_equal(x, y)
    gt_assignment = tf.logical_and(out[0], out[1])
    
    all_idx = tf.reduce_any(gt_assignment, 1)
    fg_idx  = tf.where(all_idx)
    bg_idx  = tf.where(tf.logical_not(all_idx))
    
    fg_assignment = tf.gather_nd(gt_assignment, fg_idx)
    bg_assignment = tf.gather_nd(gt_assignment, bg_idx)
    
    # Get labels and boxes for roi
    fg_labels = tf.map_fn(lambda x: foreground_labels(labels, x), fg_assignment, dtype='int32')    
    label_targets = fg_labels    
    human_targets, object_targets = tf.map_fn(lambda x: regression_boxes(human_targets, object_targets, x), fg_assignment, dtype=('float', 'float'))  
    
    label_targets = K.expand_dims(label_targets, axis=0)
    human_targets = K.expand_dims(human_targets, axis=0)
    object_targets = K.expand_dims(object_targets, axis=0)
    return human_targets

def lambda_output_shape(input_shape):
    return 2, 1, None, None

def sample_indices(indices, size):
    return indices[:size, :]
    return tf.random_shuffle(indices)[:size, :]


result = Lambda(lambda_func, output_shape=lambda_output_shape)([result, human_targets, object_targets, labels_input])

model = Model(inputs=[human_outputs, object_outputs, human_targets, object_targets, labels_input], outputs=result)

pred = model.predict([boxes1, boxes3, boxes2, boxes2, labels])[0]



