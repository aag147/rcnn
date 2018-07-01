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

import utils
import numpy as np


if True:
    # Load data
    print('Loading data...')
    data = extract_data.object_data()
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

rois_path = cfg.my_output_path + dataset + '/'
save_path = cfg.my_output_path
inputsMeta = {}

for idx, (imageID, imageMeta) in enumerate(imagesMeta.items()):
    imageName = imageMeta['imageName'].split('.')[0]
    imageID = str(imageMeta['imageID'])
    
#    imageID='339823'
#    imageName = '000000339823'
    
    roisMeta = utils.load_obj(rois_path + imageName)
    target_deltas = np.copy(roisMeta['target_deltas']).tolist()
    
    roisMeta['rois'] = [[int(x*1000) for x in box] for box in roisMeta['rois']]
    new_target_deltas = []
    for row in roisMeta['target_deltas']:
        coord = []
        for x in row:
            coord.append(int(x*1000))
        new_target_deltas.append(coord)
    roisMeta['target_deltas'] = new_target_deltas
    inputsMeta[imageID] = roisMeta
    
    if idx+1 % 1000 == 0:
        utils.update_progress_new(idx+1, nb_images, imageID)
    
utils.save_obj(inputsMeta, save_path + 'proposals_'+dataset)
