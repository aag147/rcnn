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
       filters_detection,\
       filters_helper as helper
import utils
import draw
import numpy as np
import cv2 as cv
import glob
import os
import shutil

if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels
    inv_hoi_mapping = {x['pred']+x['obj']:idx for idx,x in enumerate(hoi_mapping)}
    
    path = 'C:\\Users\\aag14/Documents/Skole/Speciale/PPMI/ori_image/'
    images_path = 'C:\\Users\\aag14/Documents/Skole/Speciale/data/TUPPMI/images/test/'

    objs = ['cello', 'flute', 'french horn', 'guitar', 'harp', 'saxophone', 'trumpet', 'violin']
    preds = ['play_instrument', 'with_instrument']
    
    imagesMeta = {}
    
    data_type = 'test'
    
    for obj in objs:
        for pred in preds:
            real_pred = 'play' if pred == 'play_instrument' else 'hold'
            for filename in glob.iglob(path + pred +'/' + obj + '/' + data_type + '/*'):
                 imageName = filename.split('\\')[-1]
                 imageID  = imageName.split('.')[0]
                 label = inv_hoi_mapping[real_pred+obj]
#                 print(imageID, pred, obj, label)
                 if imageID in imagesMeta:
                     print('whaaaaaaaat', imageID)
                 imagesMeta[imageID] = {'imageID':imageID, 'imageName':imageName, 'label':label}
#                 if not os.path.exists(images_path+imageID+'.jpg'):
#                     shutil.copy2(filename, images_path)
                    
    utils.save_dict(imagesMeta, path+data_type)