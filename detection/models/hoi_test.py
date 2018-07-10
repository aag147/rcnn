# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:26:05 2018

@author: aag14
"""
import utils,\
       metrics
import filters_detection,\
       filters_rpn,\
       filters_hoi
       
import os
import numpy as np
import sys

def saveInputData(generator, Stages, cfg):  
    cfg.my_output_path = cfg.results_path + 'det' + cfg.my_results_dir + '/output/' + generator.data_type + '/'
    
    if not os.path.exists(cfg.my_output_path):
        os.makedirs(cfg.my_output_path)
    if not os.path.exists(cfg.my_output_path):
        raise Exception('Output directory does not exist! %s' % cfg.my_output_path)
    save_path = cfg.my_output_path
    print('   save_path:', save_path)

    genIterator = generator.begin()
#    inputMeta = {}
    
    for batchidx in range(generator.nb_batches):
#        [img,proposals], y, imageMeta, imageDims, times = next(genIterator)
        X, y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageName'].split('.')[0]
        if batchidx % 1 == 0 or batchidx==100 or batchidx==250:
            utils.update_progress_new(batchidx, generator.nb_batches, imageID)
        
        
        path = save_path + imageID + '.pkl'
        if os.path.exists(path):
            continue
        
        #STAGE 1
        proposals = Stages.stageone([X], y, imageMeta, imageDims)
        
        #STAGE 2
        bboxes = Stages.stagetwo([proposals], imageMeta, imageDims)
        if bboxes is None:
            utils.save_obj(None, save_path + imageID)
            continue
        
        #STAGE 3
        all_hbboxes, all_obboxes, all_target_labels, val_map = Stages.stagethree_targets(bboxes, imageMeta, imageDims)
        if all_hbboxes is None:
            utils.save_obj(None, save_path + imageID)
            continue
        
        #CONVERT
        inputMeta = filters_hoi.convertData([all_hbboxes, all_obboxes, all_target_labels, val_map], cfg)
        
        utils.save_obj(inputMeta, save_path + imageID)


def saveEvalData(generator, Stages, cfg):
    genIterator = generator.begin()
    
    evalData = []
    for batchidx in range(generator.nb_batches):
    
        X, y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageID']
        if batchidx+1 % 100 == 0:
            utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
        
        #STAGE 1
        proposals = Stages.stageone(X, y, imageMeta, imageDims)
        
        #STAGE 2
        bboxes = Stages.stagetwo(proposals, imageMeta, imageDims)
        if bboxes is None:
            continue
        
        #STAGE 3
        hbboxes, obboxes, props = Stages.stagethree(bboxes, imageMeta, imageDims)
        if hbboxes is None:
            continue
        
        #CONVERT
        evalData += filters_hoi.convertResults(hbboxes, obboxes, props, imageMeta, imageDims['scale'], cfg.rpn_stride)
    return evalData

def saveEvalResults(evalData, generator, cfg, obj_mapping, hoi_mapping):
    path = cfg.part_results_path + "HICO/hoi" + cfg.my_results_dir
    mode = generator.data_type
    
    if not os.path.exists(path):
        path = path[:-1]
    path += '/'
    
    utils.save_dict(evalData, path+mode+'_res')
    
    mAP, AP = metrics.computeHOImAP(evalData, generator.imagesMeta, obj_mapping, hoi_mapping, cfg)
    saveMeta = {'mAP': mAP, 'AP': AP.tolist()}
    utils.save_dict(saveMeta, path+mode+'_mAP')
    print('mAP', mode, mAP)