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


np.seterr(all='raise')

#plt.close("all")

if True:
    # Load data
    print('Loading data...')
    data = data(method='normal')
    cfg = data.cfg
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)

trainIterator = genTrain.begin()


idx = 0
j = 0
for i in range(5):    
    X, y, imageMeta, img = next(trainIterator)
    j += y.shape[1]
#    utils.update_progress(i / genTest.nb_samples)

    print(X[1].shape)
    
    prsBB = X[0][idx,::]
    objBB = X[1][idx,::]
    patterns = X[2][idx,::]
    y     = y[idx]
#    image -= np.min(image)
    prsBB += cfg.PIXEL_MEANS
    prsBB = prsBB.astype(np.uint8)
    objBB += cfg.PIXEL_MEANS
    objBB = objBB.astype(np.uint8)
    draw.drawHoICrops(prsBB, objBB, patterns)


    f, spl = plt.subplots(1,1)
    spl.axis('off')
    spl.imshow(img)