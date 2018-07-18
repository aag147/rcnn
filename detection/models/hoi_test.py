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
    cfg.my_output_path = cfg.results_path + 'det' + cfg.my_results_dir + '/output/' + generator.data_type + 'newest/'
    
    if not os.path.exists(cfg.my_output_path):
        os.makedirs(cfg.my_output_path)
    if not os.path.exists(cfg.my_output_path):
        raise Exception('Output directory does not exist! %s' % cfg.my_output_path)
    save_path = cfg.my_output_path
    print('   save_path:', save_path)

    genIterator = generator.begin()
    inputMeta = None
    bboxes = None
    
    for batchidx in range(generator.nb_batches):
        X, y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageName'].split('.')[0]
#        
#        imageID = 'HICO_train2015_00019135'
#        imageMeta = generator.imagesMeta[imageID]
#        X, y, imageDims = Stages.stagezero(imageMeta, generator.data_type)
        if (batchidx+1) % (generator.nb_batches // 100) == 0 or batchidx==0 or (batchidx+1) == generator.nb_batches:
            utils.update_progress_new(batchidx, generator.nb_batches, imageID)
        
        
        path = save_path + imageID + '.pkl'
        if os.path.exists(path):
#            print(path)
            continue
        
        #STAGE 1
        proposals = Stages.stageone([X], y, imageMeta, imageDims)
        
        #STAGE 2
        bboxes = Stages.stagetwo([proposals], imageMeta, imageDims)
        
        #STAGE 3
        all_hbboxes, all_obboxes, all_target_labels, val_map = Stages.stagethree_targets(bboxes, imageMeta, imageDims)
        
        #CONVERT
        inputMeta = filters_hoi.convertData([all_hbboxes, all_obboxes, all_target_labels, val_map], cfg, mode=generator.data_type)
        
        utils.save_obj(inputMeta, save_path + imageID)
    return inputMeta, imageID, bboxes

def saveEvalData(generator, Stages, cfg, obj_mapping):
    genIterator = generator.begin()
    
    evalData = []
    for batchidx in range(generator.nb_batches):
    
        [X, all_hbboxes, all_obboxes, all_val_map], all_target_labels, imageMeta, imageDims, _ = next(genIterator)
        if X is None:
            continue
#        print(imageMeta)
        imageID = imageMeta['imageName'].split('.')[0]
        if (batchidx+1) % (generator.nb_batches // 100) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
            utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
        #STAGE 3
        pred_hbboxes, pred_obboxes, pred_props = Stages.stagethree([X,all_hbboxes,all_obboxes], imageMeta, imageDims, obj_mapping=None)
        if pred_hbboxes is None:
            continue
        
        
        #CONVERT
        evalData += filters_hoi.convertResults(pred_hbboxes, pred_obboxes, pred_props, imageMeta, imageDims['scale'], cfg.rpn_stride, obj_mapping)
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