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

import numpy as np

import utils,\
       extract_data,\
       methods,\
       losses,\
       callbacks,\
       filters_helper as helper,\
       filters_rpn,\
       filters_hoi,\
       stages
from hoi_generators import DataGenerator


# meta data
data = extract_data.object_data(False)

# config
cfg = data.cfg
#cfg.do_fast_hoi = True
obj_mapping = data.class_mapping
hoi_mapping = data.hoi_labels

# data
genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
#genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='val')


#Stages = stages.AllStages(cfg, None, obj_mapping, hoi_mapping, mode='train')
#imageID = 'HICO_test2015_00003764'
#imageMeta = genTrain.imagesMeta[imageID]
#X, y, imageDims = Stages.stagezero(imageMeta, genTrain.data_type)
#imageInputs = utils.load_obj(cfg.my_input_path + 'train/' + imageID)
#Y_tmp = filters_hoi.loadData(imageInputs, imageDims, cfg)
#hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, cfg)
#patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, cfg)
#hcrops, ocrops = filters_hoi.convertBB2Crop(X, hbboxes, obboxes, imageDims)

genItr = genTrain.begin()
for batchidx in range(genTrain.nb_batches):
    [hcrops, ocrops, patterns, hbboxes, obboxes], target_labels, imageMeta, imageDims, _ = next(genItr)
#    [img, hbboxes, obboxes, patterns], target_labels, imageMeta, imageDims, _ = next(genItr)
    
    
    X, _ = filters_rpn.prepareInputs(imageMeta, genTrain.images_path, cfg)
    imageID = imageMeta['imageName']
    utils.update_progress_new(batchidx+1, genTrain.nb_batches, imageID)
#    continue
    import draw
#    draw.drawPositiveCropHoI(None, None, hcrops, ocrops, patterns, target_labels, imageMeta, imageDims, cfg, obj_mapping)
    
#    import draw
    
    img = np.copy(X[0])
    img += cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    hbboxes = np.expand_dims(hbboxes,axis=0)
    obboxes = np.expand_dims(obboxes,axis=0)
    h_bboxes, o_bboxes = filters_hoi.unprepareInputs(hbboxes, obboxes, imageDims)
    draw.drawGTBoxes(img, imageMeta, imageDims)
#    draw.drawPositiveHoIs(img, h_bboxes[0], o_bboxes[0], target_labels, obj_mapping, imageMeta, imageDims, cfg)
    draw.drawPositiveHoI(img, h_bboxes[0], o_bboxes[0], patterns, target_labels, imageMeta, imageDims, cfg, obj_mapping)
#    
