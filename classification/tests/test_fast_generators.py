# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys

sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../layers/')
sys.path.append('../data/')
sys.path.append('../../detection/filters/')


from model_trainer import model_trainer
from load_data import data
from fast_generators import DataGenerator
from methods import Fast_HO_RCNN
import utils
import draw

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from matplotlib import pyplot as plt

import tensorflow as tf  
from models import RoiPoolingConv

import numpy as np
import cv2 as cv


np.seterr(all='raise')

#plt.close("all")

# Load data
print('Loading data...')
data = data(False)
cfg = data.cfg
cfg.fast_rcnn_config()

# Create batch generators
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train', labels=data.labels)
genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test', labels=data.labels)


trainIterator = genTest.begin()

fast_model = Fast_HO_RCNN(cfg)
#trainer = model_trainer(model=model, genTrain=genTrain, genVal=None, genTest=None, task=cfg.task)
#trainer.compileModel(cfg)


def unnormCoords(box, shape):
    xmin = box[1] * shape[1]; xmax = box[3] * shape[1]
    ymin = box[0] * shape[0]; ymax = box[2] * shape[0]
    return [xmin, ymin, xmax-xmin, ymax-ymin]    

idx = 0
j = 0
for i in range(genTest.nb_batches):
    X, y = next(trainIterator)
    break
    j += y.shape[1]
#    utils.update_progress(i / genTest.nb_samples)
    utils.update_progress_new(j, genTest.nb_samples, [y.shape[1],i,0,0], '')
#    continue
#    print('t',X[0].shape, X[1].shape, X[2].shape, y.shape)
    print(X[1].shape)
    
    image = X[0][idx]
    prsBB = X[1][idx][0]
#    objBB = X[2][idx][0]
    y     = y[idx]
    image -= np.min(image)
    image /= 255
    image = image[:,:,(2,1,0)]
#    image = np.fliplr(image)
    prsBB = unnormCoords(prsBB[1:], image.shape)
#    prsBB[0] = image.shape[1] - prsBB[0] - prsBB[2]
#    objBB = unnormCoords(objBB[1:], image.shape)
    draw.drawHOI(image, prsBB, prsBB)
    
    
    cfg.pool_size = 21
    image_input = Input(shape=(None,None,3))
    boxes_input = Input(shape=(None,5))
    newHeight = int((image.shape[0] - 11) / 4 / 4)
    newWidth = int((image.shape[1] - 11) / 4 / 4)
    image = cv.resize(image, (newHeight, newWidth))
    image = np.expand_dims(image, axis=0)
#    final_output = tf.image.crop_and_resize(image_input, boxes=boxes_input, box_ind=idxs_input, crop_size=(pool_size, pool_size))    
    roi_output = RoiPoolingConv(cfg)([image_input, boxes_input])
    model = Model(inputs=[image_input, boxes_input], outputs=roi_output)  
    
    pred = model.predict_on_batch(X)
    plt.imshow(pred[0,0,:,:,:])
    
    if i > 0:
        break