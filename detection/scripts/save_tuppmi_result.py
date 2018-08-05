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
          
        #CONVERT
        evalData = filters_hoi.convertResults(pred_hbboxes, pred_obboxes, pred_props, imageMeta, imageDims['scale'], cfg, hoi_mapping)
        utils.save_obj(evalData, save_path + imageID)
#        break
    return evalData, imageMeta

def saveEvalResults(generator, cfg, obj_mapping, hoi_mapping, evalData=None):
    evalDataInput = evalData
    my_output_path = cfg.results_path + 'hoi' + cfg.my_results_dir + '/res/' + generator.data_type + 'res/'
    
    path = cfg.part_results_path + cfg.dataset + "/hoi" + cfg.my_results_dir
    mode = generator.data_type
    
    if not os.path.exists(path):
        path = path[:-1]
    path += '/'
    nb_empty = 0
    nb_gt_samples = np.zeros(cfg.nb_hoi_classes)
    if evalData is None or True:
        evalData = []
        for batchidx, (imageID, imageMeta) in enumerate(generator.imagesMeta.items()):
            if (batchidx+1) % (max(1,generator.nb_batches // 100)) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
                utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
            
            gt_label = imageMeta['label']
            
            data = utils.load_obj(my_output_path + imageID)
            best_score = -1
            best_idx = None
            for idx, det in enumerate(data):
                if det['score'] > best_score:
                    best_score = det['score']
                    best_idx = idx
                
            best_data = [data[best_idx]]
            best_data[0]['gt'] = gt_label
            nb_gt_samples[gt_label] += 1
                
            if data is not None and len(data) > 0:
                evalData.extend(best_data)

                
    evalData = evalDataInput
    evalData = cp.copy(evalData)
#    return evalData, None
    
    cfm = np.zeros((cfg.nb_hoi_classes, cfg.nb_hoi_classes))
    
    APs = np.zeros(cfg.nb_hoi_classes)
    for label in range(cfg.nb_hoi_classes):
        subset = [x for x in evalData if x['category_id']==label]
        
        if len(subset)==0:
            continue
        
        props = [x['score'] for x in subset]
        
        nb_preds = len(props)
        idxs = np.argsort(props)[::-1]
        
        tps = np.zeros(nb_preds)
        fps = np.zeros(nb_preds)
        
        nb_class_samples = nb_gt_samples[label]
        
        for i in range(nb_preds):
            idx = idxs[i]
            pred = subset[idx]
            
            cfm[pred['gt'], pred['category_id']] += 1
            
            if pred['category_id'] == pred['gt']:
                tps[i] = 1
            else:
                fps[i] = 1
        
        if np.sum(tps)==0:
            continue
        
        tp = np.cumsum(tps)
        fp = np.cumsum(fps)
        
        print(label, tp[-1], nb_class_samples)
#        break
        recall = tp / nb_class_samples
        precision = tp / (fp+tp)
        

        Ps = np.zeros((11))
        for rec in range(0,10):
            idxs = np.where(recall>= rec/10.0)[0]
            if len(idxs) == 0:
                p = 0.0
            else:
                p = np.max(precision[idxs])
            Ps[rec] = p
                
        AP = np.mean(Ps)
        APs[label] = AP
    
    mAP = np.mean(APs)
    return evalData, mAP, cfm
    
    saveMeta = {'mAP': mAP, 'zAP': AP.tolist(), 'nb_empties': nb_empty}
    utils.save_dict(saveMeta, path+mode+'_mAP')
    print('mAP', mode, mAP)
    print('empties', nb_empty)

# Train data
#evalTest, imageMeta = saveEvalData(genTest, Stages, cfg, hoi_mapping)
allEvalData, mAP, cfm = saveEvalResults(genTest, cfg, obj_mapping, hoi_mapping, allEvalData)
