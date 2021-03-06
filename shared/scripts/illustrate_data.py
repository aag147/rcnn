# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:02:29 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../detection/models/')
sys.path.append('../../detection/filters/')
sys.path.append('../../detection/data/')
sys.path.append('../../classification/data/')


import detection.data.extract_data as extract_data
#import classification.data.load_data as extract_data
import draw
import utils

if False:
    # Load data
#    data = extract_data.data(False)
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels


trainStats = data.getLabelStats(dataset='train')
valStats = data.getLabelStats(dataset='val')
images_path = cfg.data_path + 'images/train/'

imagesMeta = data.trainGTMeta
imageIDs = list(imagesMeta.keys())
imageID  = imageIDs[1300]
imageMeta = imagesMeta[imageID]

#draw.drawObjExample(imageMeta, images_path)

draw.drawHoIExample(imageMeta, images_path, hoi_mapping)



#trainStats, trainCounts = utils.getLabelStats(data.trainGTMeta, hoi_mapping)
#valStats, valCounts = utils.getLabelStats(data.testGTMeta, hoi_mapping)

#draw.plot_object_stats(trainStats, sort=False)
#draw.plot_object_stats(valStats, sort=False)

#draw.plot_hoi_stats(trainStats, sort=False)
#draw.plot_hoi_stats(valStats, sort=False)
    
#valStats = data.getAreaStats(dataset='val')
#draw.plot_area_stats(valStats, sort=True)
#
#trainStats = data.getAreaStats(dataset='train')
#draw.plot_area_stats(trainStats, sort=False)