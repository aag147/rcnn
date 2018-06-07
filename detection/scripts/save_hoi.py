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
import metrics
import filters_helper as helper
import methods
import os
import filters_detection,\
       filters_rpn,\
       filters_hoi,\
       hoi_forward
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
    data = extract_data.object_data(False)
    cfg = data.cfg
    class_mapping = data.class_mapping
    labels = data.hoi_labels
    
#if True:
    # Create batch generators
    imagesMeta = data.testGTMeta
    genTrain = DataGenerator(imagesMeta = imagesMeta, cfg=cfg, data_type='test', do_meta=True)
    
#if True:
#    model_rpn, model_detection, model_hoi, model_all = methods.get_hoi_rcnn_models(cfg, mode='train')
    model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg, mode='test')
    if type(cfg.my_weights)==str and len(cfg.my_weights)>0:
        print('Loading my weights...')
        
        path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
        model_rpn.load_weights(path, by_name=False)
        print(model_rpn.layers[11].get_weights()[0][0,0,0,0])
        
        path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir + '/weights/' + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_detection.layers[4].get_weights()[0][0,0])
        model_detection.load_weights(path, by_name=True)
        print(model_detection.layers[4].get_weights()[0][0,0])
        
        path = cfg.part_results_path + 'HICO/hoi5c/weights/' + cfg.my_weights
        assert os.path.exists(path), 'invalid path: %s' % path
        print(model_hoi.layers[23].get_weights()[0][0,0,0,0])
        model_hoi.load_weights(path, by_name=False)
        print(model_hoi.layers[23].get_weights()[0][0,0,0,0])
    

        print(model_rpn.layers[11].get_weights()[0][0,0,0,0])

total_times = np.array([0.0,0.0])
j = 0

genIterator = genTrain.begin()

coco_res = []
for i in range(genTrain.nb_batches):

    X, y, imageMeta, imageDims, times = next(genIterator)
    utils.update_progress_new(i+1, genTrain.nb_batches, imageMeta['id'])
    
    h_bboxes, o_bboxes, hoi_labels = hoi_forward.forward_pass([X, y, imageMeta, imageDims], [model_rpn, model_detection, model_hoi], cfg, class_mapping)       

    if h_bboxes is None:
        continue
    cocoformat = helper.hoiBBoxes2COCOformat(h_bboxes, o_bboxes, hoi_labels, imageMeta, imageDims['scale'], cfg.rpn_stride)
    coco_res += cocoformat
    

mAP, AP = metrics.computeHOImAP(coco_res, imagesMeta, class_mapping, labels, cfg)

saveMeta = {'mAP': mAP, 'AP': AP.tolist()}

path = cfg.part_results_path + 'HICO/hoi5c/hoimap'
utils.save_dict(saveMeta, path)