# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:26:05 2018

@author: aag14
"""

import utils
import filters_helper as helper,\
       filters_rpn,\
       filters_detection,\
       filters_hoi
       
       
import os


def saveInputData(generator, Stages, cfg):
    genIterator = generator.begin()
    detMeta = {}
    
    if not os.path.exists(cfg.my_save_path + generator.data_type + '/'):
        os.makedirs(cfg.my_save_path + generator.data_type + '/')
    save_path = cfg.my_save_path + generator.data_type + '/'
        
    for batchidx in range(generator.nb_batches):
    
        X, y, imageMeta, imageDims, times = next(genIterator)   
        imageID = imageMeta['imageName'].split('.')[0]
        utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
        
        #STAGE 1
        proposals = Stages.stageone(X, y, imageMeta, imageDims)
        
        #STAGE 2
        proposals, target_labels, target_deltas = Stages.stagetwo(proposals, imageMeta, imageDims, include='pre')
    
        #CONVERT
        if proposals is None:
            detMeta = None
        else:
            detMeta = filters_detection.convertData([proposals, target_labels, target_deltas], cfg)
                
        utils.save_obj(detMeta, save_path + imageID)

def saveEvalData(generator, Stages, cfg, obj_mapping):
    genIterator = generator.begin()
    evalData = []
    
    for i in range(generator.nb_batches):
        X, y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageName'].split('.')[0]
        utils.update_progress_new(i+1, generator.nb_batches, imageID)
        
        #STAGE 1
        proposals = Stages.stageone(X, y, imageMeta, imageDims)
        
        #STAGE 2
        bboxes = Stages.stagetwo(proposals, imageMeta, imageDims)
        if bboxes is None:
            continue
        
        #CONVERT
        evalData += filters_detection.convertResults(bboxes, imageMeta, obj_mapping, imageDims['scale'], cfg.rpn_stride)
        
    return evalData

def saveEvalResults(evalData, generator, cfg):
    path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir + '/'
    mode = generator.data_type
    utils.save_dict(evalData, path+mode+'res')
    
