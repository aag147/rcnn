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

if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
#    hoi_mapping = data.hoi_mapping


#trainStats = data.getLabelStats(dataset='train')
#valStats = data.getLabelStats(dataset='val')

#trainStats, trainCounts = utils.getLabelStats(data.trainGTMeta, hoi_mapping)
#valStats, valCounts = utils.getLabelStats(data.testGTMeta, hoi_mapping)
    
#valStats = data.getAreaStats(dataset='val')
#draw.plot_area_stats(valStats, sort=False)

trainStats = data.getAreaStats(dataset='train')
#draw.plot_area_stats(trainStats, sort=False)