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
    cfg.my_save_path = cfg.base_path + 'results/' + cfg.dataset + '/rpn' + cfg.my_results_dir + '/detections/'
    if not os.path.exists(cfg.my_save_path):
        raise Exception('Detection directory does not exist!')
    if not os.path.exists(cfg.my_save_path + generator.data_type + '/'):
        os.makedirs(cfg.my_save_path + generator.data_type + '/')
    save_path = cfg.my_save_path + generator.data_type + '/'
    print('   save_path:', save_path)
    
    genIterator = generator.begin()
    detMeta = {}
        
    for batchidx in range(generator.nb_batches):
    
        img, y, imageMeta, imageDims, times = next(genIterator)   
        imageID = imageMeta['imageName'].split('.')[0]
        utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
        
        path = save_path + imageID + '.pkl'
        if os.path.exists(path):
            continue
        
        #STAGE 1
        proposals = Stages.stageone([img], y, imageMeta, imageDims)
        
        #STAGE 2
        proposals, target_labels, target_deltas = Stages.stagetwo_targets(proposals, imageMeta, imageDims)
    
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
        [img,proposals], y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageID']
        utils.update_progress_new(i+1, generator.nb_batches, imageID)
        
        #STAGE 1
#        proposals = Stages.stageone([img], y, imageMeta, imageDims)
        
        #STAGE 2
        bboxes = Stages.stagetwo([img,proposals], imageMeta, imageDims)
        if bboxes is None:
            continue
        
        #CONVERT
        evalData += filters_detection.convertResults(bboxes, imageMeta, obj_mapping, imageDims['scale'], cfg.rpn_stride)
        
    return evalData

def saveEvalResults(evalData, generator, cfg):
    path = cfg.part_results_path +  cfg.dataset + "/det" + cfg.my_results_dir + '/'
    mode = generator.data_type
    utils.save_dict(evalData, path+mode+'_res')
    
