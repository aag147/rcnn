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


from load_data import data
from generators import DataGenerator
import utils
import draw
from matplotlib import pyplot as plt

import numpy as np
import cv2 as cv
import keras
np.seterr(all='raise')

#plt.close("all")

if True:
    # Load data
    print('Loading data...')
    data = data(method='normal')
    cfg = data.cfg
    
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    images_path = genTrain.images_path
    imgs = []
    for imageID, imageMeta in data.trainMeta.items():
        img = cv.imread(images_path + imageMeta['imageName'])
    
        img = img.astype(np.float32, copy=False)
        im_shape = img.shape
        img = cv.resize(img, (244,244), interpolation=cv.INTER_LINEAR)
    
        img -= cfg.PIXEL_MEANS
#        im /= 255
        imgs.append(img)
        
    model = keras.applications.vgg16.VGG16(weights='imagenet')
    
    
    # Create batch generators

trainIterator = genTrain.begin()