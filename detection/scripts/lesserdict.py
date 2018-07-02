# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import numpy as np

import utils,\
       extract_data,\
       methods,\
       losses,\
       callbacks,\
       filters_helper as helper
from det_generators import DataGenerator
    

if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    obj_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=True)
#    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)


redux = {}
imageID = '487566'
redux[imageID] = genTrain.imagesInputs[imageID]



i = 0
goal = 5000

for imageID, inputMeta in genTrain.imagesInputs.items():
    redux[imageID] = inputMeta
    utils.update_progress_new(i+1, goal, imageID)

    if i == goal:
        break
    i += 1

utils.save_obj(redux, cfg.my_output_path + 'proposals_redux')