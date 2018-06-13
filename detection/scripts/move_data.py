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

from shutil import copyfile
import os
import utils



if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data(False)
    cfg = data.cfg


dataset = 'val'

if dataset == 'val':
    print('Validation data')
    imagesMeta = data.valGTMeta
elif dataset == 'test':
    print('Test data')
    imagesMeta = data.testGTMeta
else:
    print('Train data')
    imagesMeta = data.trainGTMeta

nb_images = len(imagesMeta)

from_anchor_path = cfg.part_data_path + cfg.dataset + '/anchors/'+dataset+'/'
to_anchor_path = cfg.data_path + 'anchors/'+dataset+'/'

if not os.path.exists(to_anchor_path):
    os.makedirs(to_anchor_path)

for idx, (imageID, imageMeta) in enumerate(imagesMeta.items()):
    imageName = imageMeta['imageName'].split('.')[0] + '.pkl'
    copyfile(from_anchor_path + imageName, to_anchor_path + imageName)
    utils.update_progress_new(idx+1, nb_images, imageMeta['imageName'])
    
