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


import numpy as np


np.seterr(all='raise')

#plt.close("all")

# Load data
print('Loading data...')
data = data(method='normal')
cfg = data.cfg

# Create batch generators
genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')

trainIterator = genTrain.begin()

#fast_model = Fast_HO_RCNN(cfg)
#trainer = model_trainer(model=model, genTrain=genTrain, genVal=None, genTest=None, task=cfg.task)
#trainer.compileModel(cfg)


def unnormCoords(box, shape):
    xmin = box[1] * shape[1]; xmax = box[3] * shape[1]
    ymin = box[0] * shape[0]; ymax = box[2] * shape[0]
    return np.array([xmin, ymin, xmax-xmin, ymax-ymin])

idx = 0
j = 0
for i in range(1):
    X, y = next(trainIterator)
    j += y.shape[1]
#    utils.update_progress(i / genTest.nb_samples)
#    utils.update_progress_new(j, genTrain.nb_samples, '')
#    continue
#    print('t',X[0].shape, X[1].shape, X[2].shape, y.shape)
    print(X[1].shape)
    
    prsBB = X[0][idx,::]
    objBB = X[1][idx,::]
    patterns = X[2][idx,::]
    y     = y[idx]
#    image -= np.min(image)
#    image /= 255
#    image = image[:,:,(2,1,0)]
#    image = np.fliplr(image)
#    prsBB = unnormCoords(prsBB, image.shape)
#    objBB = unnormCoords(objBB, image.shape)
#    prsBB[0] = image.shape[1] - prsBB[0] - prsBB[2]
#    objBB = unnormCoords(objBB[1:], image.shape)
    draw.drawHoICrops(prsBB, objBB, patterns)

    
    if i > 0:
        break