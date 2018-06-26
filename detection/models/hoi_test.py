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


def saveInputData(generator, Stages, cfg):
    genIterator = generator.begin()
    inputMeta = {}
    
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
        bboxes = Stages.stagetwo(proposals, imageMeta, imageDims)
        if bboxes is None:
            inputMeta[imageID] = None
            continue
        
        #STAGE 3
        all_hbboxes, all_obboxes, all_target_labels, val_map = Stages.stagethree(bboxes, imageMeta, imageDims, include='pre')
        if all_hbboxes is None:
            inputMeta[imageID] = None
            continue
        
        #CONVERT
        inputMeta[imageID] = filters_hoi.convertData([all_hbboxes[0], all_obboxes[0], all_target_labels[0], val_map[0]], cfg)
        
    utils.save_dict(inputMeta, save_path + 'hoiputs')
    return inputMeta


def saveEvalData(generator, Stages, cfg):
    genIterator = generator.begin()
    
    evalData = []
    for batchidx in range(generator.nb_batches):
    
        X, y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageID']
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