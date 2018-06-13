# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:50:43 2018

@author: aag14
"""

import sys 
sys.path.append('../shared/')
sys.path.append('../../')

path = 'C:/Users/aag14/Documents/Skole/Speciale/COCO/annotations/image_info_test2017'

import utils


annoMeta = utils.load_dict(path)['images']

testMeta = {}

for meta in annoMeta:
    imageID = str(meta['id'])
    imageName = meta['file_name']
    
    testMeta[imageID] = {'imageName': imageName}
    
utils.save_dict(testMeta, path+'meta')