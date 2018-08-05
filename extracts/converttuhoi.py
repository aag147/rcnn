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


if False:
    # Load data
    data = extract_data.object_data(False)
    cfg = data.cfg
    obj_mapping = data.class_mapping
    hoi_mapping = data.hoi_labels    

path = 'C:\\Users\\aag14/Documents/Skole/Speciale/data/TUPPMI/'
#meta_path = path+'train'
#trainMeta = utils.load_dict(meta_path)
lbl_path  = path+'labels'
tu_labels = utils.load_dict(lbl_path)

trainMeta = utils.load_dict(path+'train_objs')
testMeta = utils.load_dict(path+'val_objs')
allMeta = dict()
allMeta.update(trainMeta)
allMeta.update(testMeta)

newMeta = {}

for imageID, meta in trainMeta.items():
    break
    objs = []
    rels = []
    for idx, (_,rel) in enumerate(meta['rels'].items()):
        [lbl] = rel['labels']
        prs = rel['prsBB']
        prs['label'] = 'person'
        obj = rel['objBB']
        obj['label'] = tu_labels[lbl]['obj']
        
        objs.append(prs)
        objs.append(obj)
        rels.append([idx*2, idx*2+1, lbl])
    newMeta[imageID] = {'imageName':meta['imageName'],'imageID':imageID, 'objects':objs, 'rels':rels}
    
    
tu_obj_mapping = {'bg':0, 'person':1, 'cello':2, 'flute':3, 'french horn':4, 'guitar':5, 'harp':6, 'saxophone':7, 'trumpet':8, 'violin':9}

#utils.save_dict(tu_obj_mapping, path+'obj_mapping')
#utils.save_dict(allMeta, path+'train_objs')