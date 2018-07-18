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
import random as r

def saveInputData(generator, Stages, cfg):    
    cfg.my_output_path = cfg.results_path + 'rpn' + cfg.my_results_dir + '/output/'
    if not os.path.exists(cfg.my_output_path):
        raise Exception('Output directory does not exist! %s' % cfg.my_output_path)
    if not os.path.exists(cfg.my_output_path + generator.data_type + '/'):
        os.makedirs(cfg.my_output_path + generator.data_type + '/')
    save_path = cfg.my_output_path + generator.data_type + '/'
    print('   save_path:', save_path)
    
    genIterator = generator.begin()
    detMeta = {}
        
    for batchidx in range(generator.nb_batches):
    
        img, y, imageMeta, imageDims, times = next(genIterator)   
        imageID = str(imageMeta['imageID'])
        if (batchidx+1) % (generator.nb_batches // 100) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
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
            utils.save_obj(None, save_path + imageID)
            continue
        
        detMeta = filters_detection.convertData([proposals, target_labels, target_deltas], cfg)
                        
        utils.save_obj(detMeta, save_path + imageID)

def saveEvalData(generator, Stages, cfg, obj_mapping):
    
    cfg.my_output_path = cfg.results_path + 'det' + cfg.my_results_dir + '/res/' + generator.data_type + '/'
    
    if not os.path.exists(cfg.my_output_path):
        os.makedirs(cfg.my_output_path)
    save_path = cfg.my_output_path
    print('   save_path:', save_path)
    
    evalData = []
    
    imagesIDs = list(generator.imagesMeta.keys())
    r.shuffle(imagesIDs)
    for batchidx, imageID in enumerate(imagesIDs):    
        if (batchidx+1) % (generator.nb_batches // 100) == 0 or batchidx==1 or (batchidx+1) == generator.nb_batches:
            utils.update_progress_new(batchidx+1, generator.nb_batches, imageID)
                
        path = save_path + imageID + '.pkl'
        if os.path.exists(path):
            continue
        imageMeta = generator.imagesMeta[imageID]
        imageMeta['id'] = imageID
        
        imageInputs = generator._getImageInputs(imageID)
        X, imageDims = filters_rpn.prepareInputs(imageMeta, generator.images_path, cfg)
        Y_tmp = filters_hoi.loadData(imageInputs, imageDims, cfg)
        proposals, target_labels, target_deltas = Y_tmp
        #STAGE 1
#        proposals = Stages.stageone([img], y, imageMeta, imageDims)
        
        #STAGE 2
        bboxes = Stages.stagetwo([X,proposals], imageMeta, imageDims)
        if bboxes is None:
            continue
        
        #CONVERT
        evalData = filters_detection.convertResults(bboxes[0], imageMeta, obj_mapping, imageDims['scale'], cfg.rpn_stride)
        utils.save_obj(evalData, save_path + str(imageID))
        
    return evalData

def saveEvalResults(evalData, generator, cfg):
    
    my_output_path = cfg.results_path + 'det' + cfg.my_results_dir + '/res/' + generator.data_type + '/'

    evalData = []
    nb_empty = 0
    for batchidx, (imageID, imageMeta) in enumerate(generator.imagesMeta.items()):
        if os.path.exists(my_output_path + str(imageID)):
            evalData.append(utils.load_obj(my_output_path + imageID))
        else:
            nb_empty += 1
    
    path = cfg.results_path + "det" + cfg.my_results_dir + '/'
    mode = generator.data_type
    utils.save_dict(evalData, path+mode+'_res')
    print('nb_empty', nb_empty)
