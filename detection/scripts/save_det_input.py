# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

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

import methods,\
       stages
import det_test


if True:
    # meta data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
    #genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test') 


#if True:
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=False, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')

#    det_test.saveInputData(genVal, Stages, cfg)
    det_test.saveInputData(genTrain, Stages, cfg)
    
print()
print('Path:', cfg.my_output_path)

#import filters_rpn
#import filters_detection
#import draw
#import numpy as np
#import utils
#imageID='7088'
#imagesMeta = data.valGTMeta
#imagesInputs = utils.load_obj(cfg.my_output_path+'val/proposals')
#
#imageMeta = imagesMeta[imageID]
#
#imageInputs = imagesInputs[imageID]
#imageMeta['id'] = imageID
#
#X, imageDims = filters_rpn.prepareInputs(imageMeta, genVal.images_path, cfg)
#
#Y_tmp = filters_detection.loadData(imageInputs, cfg)
#
#bboxes, target_labels, target_deltas = filters_detection.reduceData(Y_tmp, cfg)
#
#idxs = np.where(target_labels[0,:,1:]==1)[0]
#bboxes = bboxes[:,idxs,:]
#
#labels = np.argmax(target_labels[0,:,:], axis=1)[idxs]
#labels = np.reshape(labels, (1,len(idxs),1))
#
#bboxes = np.concatenate((bboxes, labels*0+1), axis=2)
#bboxes = np.concatenate((bboxes, labels), axis=2)
#
#
#img = np.copy(X[0])
#img -= np.min(img)
#img /= np.max(img)
#
#draw.drawOverlapRois(img, bboxes[0], imageMeta, imageDims, cfg, obj_mapping)
