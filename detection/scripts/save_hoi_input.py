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

import methods,\
       stages,\
       filters_rpn,\
       filters_hoi
import hoi_test
import numpy as np
import utils

if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels    
    
#     Create batch generators
#    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True)
    

    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')

if True:
    inputMeta = hoi_test.saveInputData(genTest, Stages, cfg)

if False:
    inputMeta = utils.load_obj(cfg.my_output_path + 'hoiputs_'+genTest.data_type)
    keys = list(inputMeta.keys())
    
    for imageID in keys:
        imageMeta = data.valGTMeta[imageID]
        imageInputs = inputMeta[imageID]
    
        X, imageDims = filters_rpn.prepareInputs(imageMeta, genTest.images_path, cfg)
        Y_tmp = filters_hoi.loadData(imageInputs, imageDims, cfg)
    
        hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, cfg)
        patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, cfg)
        hcrops, ocrops = filters_hoi.convertBB2Crop(X, hbboxes, obboxes, imageDims)
    
        import draw
        img = np.copy(X[0])
        img += cfg.PIXEL_MEANS
        img = img.astype(np.uint8)
        draw.drawGTBoxes(img, imageMeta, imageDims)
        draw.drawPositiveCropHoI(hbboxes[0], obboxes[0], hcrops, ocrops, patterns[0], target_labels[0], imageMeta, imageDims, cfg, obj_mapping)

print()
print('Path:', cfg.my_output_path)
