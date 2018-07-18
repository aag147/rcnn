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
genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test', approach='new')


Stages = stages.AllStages(cfg, None, obj_mapping, hoi_mapping, mode='test')
imageID = 'HICO_test2015_00004209'
imageMeta = genTest.imagesMeta[imageID]
X, y, imageDims = Stages.stagezero(imageMeta, genTest.data_type)
imageInputs = utils.load_obj(cfg.my_input_path + 'testnew/' + imageID)
Y_tmp = filters_hoi.loadData(imageInputs, imageDims, cfg)
[hbboxes, obboxes, target_labels, all_val_map] = Y_tmp
#hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, cfg)
patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, cfg)
#hcrops, ocrops = filters_hoi.convertBB2Crop(X, hbboxes, obboxes, imageDims)


#Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=False)
#cfg.det_nms_overlap_thresh = 0.5
#Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='train')
#
##STAGE 0
#X, y, imageDims = Stages.stagezero(imageMeta, 'test')
#
##STAGE 1
#proposals = Stages.stageone([X], y, imageMeta, imageDims)
#
##STAGE 2
#bboxes = Stages.stagetwo([proposals], imageMeta, imageDims)
#
##STAGE 3
#Stages = stages.AllStages(cfg, None, obj_mapping, hoi_mapping, mode='test')
#hbboxes, obboxes, target_labels, val_map = Stages.stagethree_targets(bboxes, imageMeta, imageDims)
#patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, cfg)
#CONVERT
#inputMeta = filters_hoi.convertData([hbboxes, obboxes, target_labels, val_map], cfg, mode=genTest.data_type)
#
#utils.save_obj(inputMeta, cfg.part_data_path + imageID)

genItr = genTest.begin()
for batchidx in range(genTest.nb_batches):
    break
    [img, all_hbboxes, all_obboxes, all_val_map], target_labels, imageMeta, imageDims, _ = next(genItr)
    utils.update_progress_new(batchidx+1, genTest.nb_batches, imageID)
    continue
    
#    [img, hbboxes, obboxes, patterns], target_labels, imageMeta, imageDims, _ = next(genItr)
    
    X, _ = filters_rpn.prepareInputs(imageMeta, genTest.images_path, cfg)
    imageID = imageMeta['imageName']
    
#    continue
#    draw.drawPositiveCropHoI(None, None, hcrops, ocrops, patterns, target_labels, imageMeta, imageDims, cfg, obj_mapping)
    
#    import draw
import draw  
img = np.copy(X[0])
img += cfg.PIXEL_MEANS
img = img.astype(np.uint8)
#hbboxes = np.expand_dims(hbboxes,axis=0)
#obboxes = np.expand_dims(obboxes,axis=0)
h_bboxes, o_bboxes = filters_hoi.unprepareInputs(hbboxes, obboxes, imageDims)
draw.drawGTBoxes(img, imageMeta, imageDims)
draw.drawPositiveHoIs(img, hbboxes[0], obboxes[0], target_labels[0], hoi_mapping, imageMeta, imageDims, cfg)
draw.drawPositiveHoI(img, hbboxes[0], obboxes[0], patterns[0], target_labels[0], imageMeta, imageDims, cfg, obj_mapping)
#draw.drawOverlapRois(img, bboxes[0], imageMeta, imageDims, cfg, obj_mapping)
#draw.drawHumanAndObjectRois(img, bboxes[0])