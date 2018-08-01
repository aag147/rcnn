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
       filters_hoi
import hoi_test
import os
import utils
import random as r
import copy as cp
import numpy as np

if True:
    # Load data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True, mode='test')
    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test')

    
    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=True)
    Stages = stages.AllStages(cfg, Models, obj_mapping, hoi_mapping, mode='test')
    
sys.stdout.flush()
# Test data
#evalTest, imageMeta = hoi_test.saveEvalData(genTest, Stages, cfg, hoi_mapping)
#hoi_test.saveEvalResults(genTest, cfg, obj_mapping, hoi_mapping)
#import draw
#draw.drawOverlapHOIRes(evalTestSub, genTest.imagesMeta, obj_mapping, hoi_mapping, genTest.images_path)


def saveEvalData(generator, Stages, cfg, hoi_mapping):
    cfg.my_output_path = cfg.results_path + 'hoi' + cfg.my_results_dir + '/res/' + generator.data_type + 'res/'
    
    if not os.path.exists(cfg.my_output_path):
        os.makedirs(cfg.my_output_path)
    save_path = cfg.my_output_path
    print('   save_path:', save_path)
    
    evalData = []
    imageMeta = None
    imagesIDs = list(generator.imagesMeta.keys())
    r.shuffle(imagesIDs)
    for batchidx, imageID in enumerate(imagesIDs):        
        if (batchidx+1) % (max(100,generator.nb_batches // 100)) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
            utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
                
        path = save_path + imageID + '.pkl'
        if os.path.exists(path):
            continue
        
        imageMeta = generator.imagesMeta[imageID]
        imageMeta['id'] = imageID
        X, y, imageDims = Stages.stagezero(imageMeta, generator.data_type)

#        gt_label = imageMeta['label']
#        intval_beg = gt_label // 2 * 2
        
        proposals = Stages.stageone([X], y, imageMeta, imageDims)
        bboxes = Stages.stagetwo([proposals], imageMeta, imageDims)
        pred_hbboxes, pred_obboxes, pred_props = Stages.stagethree([bboxes], imageMeta, imageDims)
        
#        new_pred_props = np.zeros_like(pred_props)
#        max_pred_idx = None
#        max_pred = 0
#        for pred_idx in range(pred_props.shape[0]):
#            label = np.argmax(pred_props[pred_idx,intval_beg:intval_beg+1])
#            pred = pred_props[pred_idx,label]
#            new_pred_props[pred_idx,label] = pred
#            if pred > max_pred:
#                max_pred = pred
#                max_pred_idx = pred_idx
                
#        pred_hbboxes = pred_hbboxes[max_pred_idx,:]
#        pred_obboxes = pred_obboxes[max_pred_idx,:]
#        pred_props = pred_props[max_pred_idx,:]
          
        #CONVERT
        evalData = filters_hoi.convertResults(pred_hbboxes, pred_obboxes, pred_props, imageMeta, imageDims['scale'], cfg, hoi_mapping)
        utils.save_obj(evalData, save_path + imageID)
#        break
    return evalData, imageMeta

def saveEvalResults(generator, cfg, obj_mapping, hoi_mapping, evalData=None):
    
    my_output_path = cfg.results_path + 'hoi' + cfg.my_results_dir + '/res/' + generator.data_type + 'res/'
    
    path = cfg.part_results_path + cfg.dataset + "/hoi" + cfg.my_results_dir
    mode = generator.data_type
    
    if not os.path.exists(path):
        path = path[:-1]
    path += '/'
    nb_empty = 0
    if evalData is None:
        evalData = []
        for batchidx, (imageID, imageMeta) in enumerate(generator.imagesMeta.items()):
            if (batchidx+1) % (max(1,generator.nb_batches // 100)) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
                utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
            print(my_output_path + imageID + '.pkl')
            if os.path.exists(my_output_path + imageID + '.pkl'):
                data = utils.load_obj(my_output_path + imageID)
                best_score = 0
                best_idx = None
                for idx, det in enumerate(data):
                    if det['score'] > best_score:
                        best_score = det['score']
                        best_idx = idx
                    
                best_data = [data[best_idx]]
                    
                if data is not None and len(data) > 0:
                    evalData.extend(best_data)
            else:
                nb_empty += 1
                
    evalData = cp.copy(evalData)
    
    
    return evalData
    
    saveMeta = {'mAP': mAP, 'zAP': AP.tolist(), 'nb_empties': nb_empty}
    utils.save_dict(saveMeta, path+mode+'_mAP')
    print('mAP', mode, mAP)
    print('empties', nb_empty)

# Train data
#evalTest, imageMeta = saveEvalData(genTest, Stages, cfg, hoi_mapping)
allEvalData = saveEvalResults(genTest, cfg, obj_mapping, hoi_mapping)
