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
import filters_helper as helper,\
       filters_rpn,\
       filters_hoi

import hoi_test
import numpy as np
import utils
import cv2 as cv

if True:
    # Load data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels    
    
#     Create batch generators
#    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
    genTest = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='test', do_meta=True)
    
    
    generator = genTest
    print(cfg.my_input_path+'testnew/')

    sys.stdout.flush()

#if True:
    nb_total = np.zeros(cfg.nb_hoi_classes)
    nb_tp = np.zeros(cfg.nb_hoi_classes)
    nb_empty = 0
    nb_total_preds = 0
    for j, (imageID, imageMeta) in enumerate(generator.imagesMeta.items()):
        
#        imageID = 'HICO_train2015_00028567'
#        imageMeta = generator.imagesMeta[imageID]
        
        if (j+1) % 100 == 0:
            utils.update_progress_new((j+1), generator.nb_images, imageID)
        
        img = cv.imread(generator.images_path + imageMeta['imageName'])
        X, imageDims = filters_rpn.prepareInputs(imageMeta, generator.images_path, cfg)
        objs = imageMeta['objects']
        gt_rels = imageMeta['rels']
        gtbboxes = helper._transformGTBBox(objs, obj_mapping, None, scale=imageDims['scale'], rpn_stride=cfg.rpn_stride, dosplit=False)
        checks = np.zeros(len(gt_rels))
        
        if np.max(gtbboxes[:,2]) > 2+imageDims['output_shape'][1] or np.max(gtbboxes[:,3]) > 2+imageDims['output_shape'][0]:
            print('bad bbs', imageID, np.max(gtbboxes[:,2]), np.max(gtbboxes[:,3]), imageDims['output_shape'])
        
        imageInputs = utils.load_obj(cfg.my_input_path + 'testnew/' + imageID)
        
        if imageInputs is None:
            idxs = []
            nb_empty += 1
        else:
            idxs = np.where(np.array(imageInputs['val_map'])==3)[0]
            hbboxes = np.array(imageInputs['hbboxes'])[idxs,:] / 1000.0
            obboxes = np.array(imageInputs['o_bboxes'])[idxs,:] / 1000.0
            labels = np.array(imageInputs['hoi_labels'])[idxs]
        
        nb_preds = len(idxs)
        
        for i in range(nb_preds):
            hbbox = np.copy(hbboxes[i,:4])
            hbbox[2] += hbbox[0]
            hbbox[3] += hbbox[1]
            obbox = np.copy(obboxes[i,:4])
            obbox[2] += obbox[0]
            obbox[3] += obbox[1]
            label = labels[i]
            
            nb_total_preds += 1
            
            gth_ols = helper._computeIoUs(hbbox, gtbboxes)
            gto_ols = helper._computeIoUs(obbox, gtbboxes)
#            print(gth_ols)
#            print(gto_ols)
            for gtidx, rel in enumerate(gt_rels):
                if checks[gtidx]:
                    continue
                if gth_ols[rel[0]] >= 0.5 and gto_ols[rel[1]] >= 0.5 and rel[2] in label:
                    checks[gtidx] = 1
                    nb_tp[rel[2]] += 1
#        print(checks, label, imageID)
                    
        for rel in gt_rels:
            nb_total[rel[2]] += 1

         
        continue
        import draw
        
        img = np.copy(X[0])
        img += cfg.PIXEL_MEANS
        img = img.astype(np.uint8)
        draw.drawGTBoxes(img, imageMeta, imageDims)    
        
        
        Y_tmp = filters_hoi.loadData(imageInputs, imageDims, cfg)
    
        hbboxes, obboxes, target_labels, val_map = Y_tmp
#        hbboxes, obboxes, target_labels, val_map = filters_hoi.reduceTargets(Y_tmp, cfg)
        patterns = filters_hoi.createInteractionPatterns(hbboxes, obboxes, cfg)
        draw.drawPositiveHoI(img, hbboxes[0], obboxes[0], None, target_labels[0], imageMeta, imageDims, cfg, obj_mapping)
        hcrops, ocrops = filters_hoi.convertBB2Crop(X, hbboxes, obboxes, imageDims)
    
#        draw.drawPositiveCropHoI(hbboxes[0], obboxes[0], hcrops, ocrops, patterns[0], target_labels[0], imageMeta, imageDims, cfg, obj_mapping)
            
        
        break
        if j == 5:
            break

    res = np.array([nb_tp[i] / nb_total[i] if nb_tp[i]>0 else 0 for i in range(cfg.nb_hoi_classes)])
    rare_idxs = np.where(nb_total<10)[0]
    unrare_idxs = np.where(nb_total>=10)[0]
    print('all', np.mean(res))
    print('rare', np.mean(res[rare_idxs]))
    print('unrare', np.mean(res[unrare_idxs]))
    print('nulls', nb_empty)