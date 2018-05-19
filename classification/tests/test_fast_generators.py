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

import numpy as np


np.seterr(all='raise')

#plt.close("all")

# Load data
print('Loading data...')
data = data()
cfg = data.cfg
cfg.fast_rcnn_config()

# Create batch generators
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, labels = data.labels, cfg=cfg, data_type='train')


trainIterator = genTrain.begin()

def unnormCoords(box, shape):
    xmin = box[1] * shape[1]; xmax = box[3] * shape[1]
    ymin = box[0] * shape[0]; ymax = box[2] * shape[0]
    return [xmin, ymin, xmax-xmin, ymax-ymin]    

idx = 0
for i in range(genTrain.nb_batches):
    X, y = next(trainIterator)
#    utils.update_progress(i / genTrain.nb_images)
    print('t',X[0].shape, X[1].shape, X[3].shape, y.shape)
    
    image = X[0][idx]
    prsBB = X[1][idx][0]
    objBB = X[2][idx][0]
    y     = y[idx]
    image -= np.min(image)
    image /= 255
    image = image[:,:,(2,1,0)]
    prsBB = unnormCoords(prsBB[1:], image.shape)
    objBB = unnormCoords(objBB[1:], image.shape)
    draw.drawHOI(image, prsBB, objBB)
    
    if i > 100:
        break