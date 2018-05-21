# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""

import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')


import extract_data
from detection_generators import DataGenerator

import numpy as np
import utils
import time
#import draw

np.seterr(all='raise')

#plt.close("all")

# Load data
print('Loading data...')
data = extract_data.object_data(False)
cfg = data.cfg
cfg.fast_rcnn_config()


#new_trainGTMeta = {}
#for ID, meta in data.trainGTMeta.items():
#    if ID == '550395':
#        new_rels = []
#        for idx, rel in enumerate(meta['objects']):
#            if idx==14:
#                continue
#            else:
#                new_rels.append(rel)
#        new_trainGTMeta[ID] = {'imageName':meta['imageName'], 'objects': new_rels}
#    else:
#        new_trainGTMeta[ID] = meta
#
#
#
#utils.save_dict(new_trainGTMeta, cfg.data_path + 'train')
# Create batch generators
genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')


trainIterator = genVal.begin()

total_times = np.array([0.0,0.0])
#j = 0
for i in range(genVal.nb_batches):
    X, y, imageMeta, imageDims, times = next(trainIterator)
 
    total_times += times
    utils.update_progress_new(i, genVal.nb_batches, list(times) + [0,0], imageMeta['imageName'])
    
#    img = X[0]
#    img -= np.min(img)
#    img /= 255
#    img = img[:,:,(2,1,0)]
#    draw.drawBoxes(img, imageMeta['objects'], imageDims)
#    print('t',X[0].shape, X[1].shape, y[0].shape, y[1].shape)
#    
    utils.save_obj(y, cfg.data_path +'anchors/val/' + imageMeta['imageName'].split('.')[0])
#    s = time.time()
#    utils.load_obj(cfg.data_path +'anchors/val/' + imageMeta['imageName'].split('.')[0])
#    f = time.time()
#    print(f-s, times[1])
#    if j > 10:
#        break
#    j += 1
#    break
#print(f-s)

