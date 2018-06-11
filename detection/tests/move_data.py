# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')


import extract_data

import numpy as np
import utils
import time
import cv2 as cv
import copy as cp
import os

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg


imagesMeta = data.trainGTMeta
nb_images = len(imagesMeta)

images_path = cfg.data_path + 'images/train/'
anchors_path = cfg.data_path + 'anchors/train/'

print('images path', images_path)
print('anchors path', anchors_path)

from_anchor_path = cfg.part_data_path + cfg.dataset + '/anchors/train/'
to_anchor_path = cfg.data_path + 'anchors/train/'

from shutil import copyfile
for imageID, imageMeta in imagesMeta.items():
    imageName = imageMeta['imageName'].split('.')[0] + '.pkl'
    copyfile(from_anchor_path + imageName, to_anchor_path + imageName)
    
