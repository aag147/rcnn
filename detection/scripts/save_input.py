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
from rpn_generators import DataGenerator
import filters_helper as helper,\
       filters_rpn

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
alltimes = np.zeros((nb_images, 4))

for batchidx, (imageID, imageMeta) in enumerate(imagesMeta.items()):
    path = anchors_path + imageMeta['imageName'].split('.')[0] + '.pkl'
    if os.path.exists(path):
        continue
    
    io_start = time.time()
    img, imageDims = filters_rpn.prepareInputs(imageMeta, images_path, cfg)
    io_end = time.time()
    pp_start = time.time()
    [Y1,Y2,M] = filters_rpn.prepareTargets(imageMeta, imageDims, cfg)
    pp_end = time.time()
    
    utils.update_progress_new(batchidx+1, nb_images, imageMeta['imageName'])
    
    times = [io_end-io_start, pp_end-pp_start] + [0,0]
    alltimes[batchidx,:] = times
    
    target_labels, target_deltas, val_map = helper.bboxes2RPNformat(Y1, Y2, M, cfg)
    rpnMeta = {'target_labels': target_labels, 'target_deltas': target_deltas, 'val_map': val_map}
    utils.save_obj(rpnMeta, cfg.data_path +'anchors/train/' + imageMeta['imageName'].split('.')[0])

print('Times', np.mean(alltimes, axis=0))
