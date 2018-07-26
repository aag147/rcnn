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

import utils
import numpy as np
import filters_helper as helper
import os
from hoi_generators import DataGenerator
import metrics
import draw

if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels


if False:
    path = cfg.part_results_path + 'HICO/hoi80fastvgg2/history.txt'
    hist =  np.loadtxt(path, delimiter=', ')
    hist[:,0] /= 2

    import filters_rpn
    draw.plotRPNLosses(hist, mode='hoi', yaxis='log')
    

def loadEvalData(generator, my_output_path):    
    evalData = []
    for batchidx, (imageID, imageMeta) in enumerate(generator.imagesMeta.items()):
        if (batchidx+1) % (max(1,generator.nb_batches // 100)) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
            utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
        
        data = utils.load_obj(my_output_path + imageID)
        if data is not None and len(data) > 0:
            evalData.extend(data)

    return evalData

if False:
    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test', approach='new')
    my_output_path = cfg.part_results_path + 'HICO/hoi80fastvgg2/res/testnew/'
    evalData = loadEvalData(genTest, my_output_path)

if False:
    evalDataRdx = []
    overlap_thresh = 0.05
    for line in evalData:
        score = line['score']
        if score >= overlap_thresh:
            evalDataRdx.append(line)
    mAPRdx, AP_mapRdx, evalsRdx = metrics.computeHOImAP(evalDataRdx, imagesMeta, obj_mapping, hoi_mapping, cfg, do_cfm=True)    
    
if True:

    cfm = np.zeros((601,601))
    for line in evalsRdx:
        pred = line['category_id']+1
        gt   = line['eval']+1
        cfm[gt,pred] += 1
        
    cfm_norm = cfm / (np.sum(cfm,axis=0)+0.0000000001)
    draw.plot_confusion_matrix(cfm, normalize=True, no_bg=False)
    draw.plot_confusion_matrix(cfm, normalize=True, no_bg=True)

if False:
    images_path = cfg.data_path + 'images/val/'
        
    path = cfg.my_results_path + 'val_res01'
    preds =  utils.load_dict(path)
    imagesMeta = data.valGTMeta
    
    predsMeta = {}
    for pred in preds:
        imageID = pred['image_id']
        if imageID not in predsMeta:
            predsMeta[imageID] = []
        predsMeta[imageID].append(pred)
        
        
    cfm = np.zeros((81,81))
    
    
    coco_mapping = helper.getCOCOMapping()
    inv_coco_mapping = {idx:label for label,idx in coco_mapping.items()}
    
    for i, (imageID, predMeta) in enumerate(predsMeta.items()):
        imageMeta = imagesMeta[str(imageID)]
        gt_bboxes = imageMeta['objects']
        gtbboxes = helper._transformGTBBox(gt_bboxes, obj_mapping, None, scale=[1,1], rpn_stride=1, dosplit=False)
        
        X, imageDims = filters_rpn.prepareInputs(imageMeta, images_path, cfg)
        
        pred_bboxes = np.ones((len(predMeta),6))
        
        utils.update_progress_new(i+1, len(predsMeta), imageID)
        
        for idx, pred in enumerate(predMeta):
            prop = pred['score']
            lbl  = obj_mapping[inv_coco_mapping[int(pred['category_id'])]]
            bbox = pred['bbox']
            pred_bboxes[idx,:] = np.copy([x*imageDims['scale'][0]/16 for x in bbox]+[prop]+[lbl])
            
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            
            if gtbboxes.shape[0]>0:
                ious = helper._computeIoUs(bbox, gtbboxes)
            
                best_iou = np.max(ious)
                best_idx = np.argmax(ious)
                best_lbl = int(gtbboxes[best_idx,4])
            else:
                best_iou = 0.0
            
            if best_iou > 0.5:
                cfm[best_lbl,lbl] += 1
            else:
                cfm[0,lbl] += 1
        
        continue
        img = np.copy(X[0])
        img += cfg.PIXEL_MEANS
        img = img.astype(np.uint8)
        
        print(imageID)
        draw.drawGTBoxes(img, imageMeta, imageDims)
        draw.drawAnchors(img, pred_bboxes, cfg)
        draw.drawOverlapRois(img, pred_bboxes, imageMeta, imageDims, cfg, obj_mapping)
        if i == 3:
            break