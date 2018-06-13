# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:25:21 2018

@author: aag14
"""

import utils
import filters_helper as helper,\
       filters_rpn,\
       filters_detection,\
       filters_hoi
       
       
import os


def saveInputData(imagesMeta, data_type, cfg):
    load_path = cfg.data_path +'images/' + data_type + '/'
    save_path = cfg.my_save_path + data_type + '/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
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