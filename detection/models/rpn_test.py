# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:25:21 2018

@author: aag14
"""

import utils,\
       metrics
       
import filters_helper as helper,\
       filters_rpn,\
       filters_detection,\
       filters_hoi
       
       
import os


def saveInputData(imagesMeta, data_type, cfg):
    load_path = cfg.data_path +'images/' + data_type + '/'
    save_path = cfg.my_save_path + data_type + 'new/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path', save_path)
        
    nb_images = len(imagesMeta)
    
    for batchidx, (imageID, imageMeta) in enumerate(imagesMeta.items()):
        imageID = imageMeta['imageName'].split('.')[0]
        utils.update_progress_new(batchidx+1, nb_images, imageID)
        
        path = save_path + imageID + '.pkl'
        if os.path.exists(path):
            continue
        
        img, imageDims = filters_rpn.prepareInputs(imageMeta, load_path, cfg)
        [Y1,Y2,M] = filters_rpn.createTargets(imageMeta, imageDims, cfg)
                        
        rpnMeta = filters_rpn.convertData([Y1, Y2, M], cfg)
        utils.save_obj(rpnMeta, save_path + imageID)
        
def saveEvalData(generator, Stages, cfg):
    genIterator = generator.begin()
    evalData = []
    
    for i in range(generator.nb_batches):
        [img,proposals], y, imageMeta, imageDims, times = next(genIterator)
        imageID = imageMeta['imageName'].split('.')[0]
        utils.update_progress_new(i+1, generator.nb_batches, imageID)
        
        #STAGE 1
#        proposals = Stages.stageone(X, y, imageMeta, imageDims)
        proposals = proposals[0,:,1:]
        
        #CONVERT
        evalData += filters_rpn.convertResults(proposals, imageMeta, imageDims['shape'])
        break
        
    return evalData

def saveEvalResults(evalData, generator, cfg, obj_mapping):
    save_path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir + '/'
    mode = generator.data_type
    
    if not os.path.exists(save_path):
        save_path = save_path[:-1]
    save_path += '/'
    
    utils.save_dict(evalData, save_path+mode+'_res')
    
    AR, R5, IoU = metrics.computeRPNAR(evalData, generator.imagesMeta, obj_mapping, cfg)
    saveMeta = {'AR': AR, 'R5': R5, 'IoU': IoU.tolist()}
    utils.save_dict(saveMeta, save_path+mode+'_mAP')
    print('R5', mode, R5)
    print('AR', mode, AR)
    return IoU