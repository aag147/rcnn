# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:52:48 2018

@author: aag14
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')


import extract_data
from detection_generators import DataGenerator

import numpy as np
import utils
import time
import draw
import filters_helper as helper
import methods
import os
import filters_detection
import cv2 as cv

np.seterr(all='raise')

#plt.close("all")


if False:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    cfg.fast_rcnn_config()
    
    # Create batch generators
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    
    images_path = cfg.data_path + 'images/'
    images_path = images_path + 'train/'
    
    images = np.zeros([65014])
    
    for imageID, imageMeta in data.trainGTMeta.items():
        img = cv.imread(images_path + imageMeta['imageName'])
        