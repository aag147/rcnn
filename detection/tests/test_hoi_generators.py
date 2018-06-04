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
from hoi_generators import DataGenerator

import numpy as np
import utils
import time
import draw
import filters_helper as helper
import methods
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


if False:
    # Load data
    print('Loading data...')
    data = extract_data.object_data(False)
    cfg = data.cfg
    class_mapping = data.class_mapping
    labels = data.hoi_labels
    
#if True:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    
#if True:
    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg, mode='train')
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg, mode='test')
    if type(cfg.my_weights)==str and len(cfg.my_weights)>0:
        print('Loading my weights...')
        path = cfg.my_weights_path + cfg.my_weights
        path = cfg.part_results_path + "COCO/" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        print(path)
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_hoi.layers[19].get_weights()[0][0,0,0,0])
        before = model_hoi.get_weights()
        model_hoi.load_weights(path, by_name=False)
        print(model_hoi.layers[19].get_weights()[0][0,0,0,0])
        after = model_hoi.get_weights()


total_times = np.array([0.0,0.0])
j = 0

genIterator = genTrain.begin()



#final_hbbs, final_obbs, final_labels, final_vals = filters_hoi.prepareTargets(boxes_nms, imageMeta, imageDims, cfg, class_mapping)

for i in range(1):
#    [X, hbb, obb, ip], target_labels, imageMeta, imageDims, times = next(genIterator)
    h_bboxes, o_bboxes = filters_hoi.unprepareInputs(hbb, obb, imageDims)
    newh_bboxes, newo_bboxes = filters_hoi.prepareInputs(h_bboxes, o_bboxes, imageDims)
    h_bboxes, o_bboxes = filters_hoi.unprepareInputs(newh_bboxes, newo_bboxes, imageDims)
    print(imageMeta['imageName'])
        
    draw.drawGTBoxes((X[0]+1.0)/2.0, imageMeta, imageDims)
    for hoiidx in range(12):
        draw.drawHoIComplete((X[0]+1.0)/2.0, h_bboxes[hoiidx,:], o_bboxes[hoiidx,:], ip[0,hoiidx,::], target_labels[0,hoiidx,:], labels, cfg)


