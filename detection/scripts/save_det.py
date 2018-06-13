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

import numpy as np
import utils
import time
import filters_helper as helper
import methods,\
       stages
import os
import filters_detection,\
       filters_rpn,\
       filters_hoi
import cv2 as cv
import copy as cp
import math

from keras.models import Sequential, Model
import h5py

np.seterr(all='raise')

#plt.close("all")


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)
    
    if not os.path.exists(cfg.my_save_path):
        os.makedirs(cfg.my_save_path)
    print('results path', cfg.my_save_path)

if True:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')

genIterator = genVal.begin()
detMeta = {}

for i in range(genVal.nb_batches):
    X, y, imageMeta, imageDims, times = next(genIterator)
    imageID = imageMeta['imageName'].split('.')[0]
    utils.update_progress_new(i+1, genVal.nb_batches, imageMeta['id'])
    
    #STAGE 1
    proposals = Stages.stageone(X, y, imageMeta, imageDims)
    
    #STAGE 2
    bboxes = Stages.stagetwo(proposals, imageMeta, imageDims)
    if bboxes is None:
        detMeta[imageID] = None
        continue
    
    #STAGE 3
    all_hbboxes, all_obboxes, all_target_labels, val_map = Stages.stagethree(bboxes, imageMeta, imageDims, include='pre')
    if all_hbboxes is None:
        detMeta[imageID] = None
        continue
    
    #CONVERT
    detMeta[imageID] = filters_hoi.convertData([all_hbboxes[0], all_obboxes[0], all_target_labels[0], val_map[0]], cfg)
    utils.update_progress_new(i+1, genVal.nb_batches, imageMeta['id'])

path = cfg.my_save_path + 'hoiputs'
utils.save_dict(detMeta, path)
print()
print('Path:', cfg.my_save_path)
