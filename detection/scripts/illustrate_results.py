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
    submodel = 'det'
    model = '_256'
    ddir = 'HICO' if submodel == 'hoi' else 'COCO'
    path = cfg.part_results_path + ddir + '/'+submodel+'80'+model+'/history.txt'
    hist =  np.loadtxt(path, delimiter=', ')
    if 'HICO' in path and hist.shape[0]>30:
        hist[:,0] /= 2

    import filters_rpn
    draw.plotRPNLosses(hist, mode=submodel)


if False:
    
    hists = []
    models = ['_256', 'S', 'SH']
    submodel = 'hoi'
    ddir = 'HICO' if submodel == 'hoi' else 'COCO'
    for i in range(3):
        path = cfg.part_results_path + ddir + '/'+submodel+'80'+models[i]+'/history.txt'
        hist =  np.loadtxt(path, delimiter=', ')
        if 'HICO' in path and hist.shape[0]>30:
            hist[:,0] /= 2
        hists.append(hist)

    draw.plotFasterLosses(hists, mode=submodel)
    
if False:
    # plot GT hoi examples
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True, mode='test', approach='new')
    imagesID = genTrain.dataID
    idxs = [59,100,222,777]
    for idx in idxs:
        imageMeta = genTrain.imagesMeta[imagesID[idx]]
        print(hoi_mapping[imageMeta['rels'][0][2]])
        draw.drawHoIExample(imageMeta, genTrain.images_path, hoi_mapping)
    

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
    # number of GT samples
    nb_gt_samples = np.zeros((cfg.nb_hoi_classes))
    for imageID, imageMeta in data.trainGTMeta.items():
        for rel in imageMeta['rels']:
            nb_gt_samples[rel[2]] += 1

if False:
    # HOI eval data
    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True, mode='test', approach='new')
    my_output_path = cfg.part_results_path + 'HICO/hoi80vggcs/res/testnew/'
    evalData = loadEvalData(genTest, my_output_path)
    imagesMeta = genTest.imagesMeta

#if False:
    # eval data with evaluation
    evalDataRdx = []
    overlap_thresh = 0.05
    for line in evalData:
        score = line['score']
        if score >= overlap_thresh:
            evalDataRdx.append(line)
    mAP, AP_map, evals = metrics.computeHOImAP(evalData, imagesMeta, obj_mapping, hoi_mapping, cfg, return_data='plt')    
    
    unrare_idxs = np.where(nb_gt_samples>=10)[0]
    rare_idxs = np.where(nb_gt_samples<10)[0]
    mAPRare = np.mean(AP_map[rare_idxs])
    mAPUnRare = np.mean(AP_map[unrare_idxs])

    
if False:
    # Multi label confusion matrix
    cfm = np.zeros((601,601))
    for line in evals:
        pred = line['category_id']+1
        gt   = line['eval']+1
        cfm[gt,pred] += 1
        
    cfm_norm = cfm / (np.sum(cfm,axis=0)+0.0000000001)
    draw.plot_confusion_matrix(cfm, normalize=True, no_bg=False)
    draw.plot_confusion_matrix(cfm, normalize=True, no_bg=True)

if False:
    # plot multi label APs
    obj_AP_map = np.zeros((80,))
    obj_hoi_nb = np.zeros((80,))
    old_obj = ''
    for idx, label in enumerate(hoi_mapping):
        obj = label['obj']
        if obj != old_obj:
            old_obj = obj
            obj_idx = obj_mapping[obj]-1
        obj_hoi_nb[obj_idx] += 1
        obj_AP_map[obj_idx] += AP_map[idx]
        
    obj_AP_map /= obj_hoi_nb 
    
    draw.pltAPs(AP_map)
    draw.pltAPs(obj_AP_map)

if False:
    # plot eval data
    props = [x['score'] for x in allEvalData]
    idxs = np.argsort(props)[::-1]
    images_path = 'C:\\Users\\aag14/Documents/Skole/Speciale/data/TUPPMI/images/test/'
    
    nb_preds = 0
    used_objs = []
    used_hois = []
    show_lines = []
    for idx in idxs:
        line = allEvalData[idx]
        pred_eval = line['gt']==line['category_id']
        if pred_eval == 1:
            continue
        if line['category_id'] in used_hois:
            continue
#        if hoi_mapping[line['category_id']]['obj'] != 'book':
#            continue
#        if line['category_id'] != 249:
#            continue
#        used_objs.append(hoi_mapping[line['category_id']]['obj'])
        used_hois.append(line['category_id'])
        nb_preds += 1
        if nb_preds < 0:
            continue
#        draw.drawHoIExample(imagesMeta[line['image_id']], images_path, hoi_mapping)
        show_lines.append(line)
#        if nb_preds == 20:
#            break
        
    draw.drawOverlapHOIRes(show_lines, genTest.imagesMeta, obj_mapping, hoi_mapping, images_path)

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