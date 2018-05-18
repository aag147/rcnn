# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:26:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

from config import basic_config
import groundtruths
import numpy as np

cfg = basic_config(False)
cfg.nb_hoi_rois = 16
cfg.nb_hoi_classes = 4
cfg.hoi_max_overlap = 0.5
cfg.hoi_min_overlap = 0.1
cfg.nb_hoi_positives = 2
cfg.nb_hoi_negatives1 = 6
cfg.nb_hoi_negatives2 = 8
cfg.rpn_stride = 16

imageDims = {'shape':(100,100,3), 'scale':[1,1]}

hbboxes = [{'xmin':0, 'xmax':20, 'ymin':0, 'ymax':100, 'label':1}, 
           {'xmin':70, 'xmax':90, 'ymin':20, 'ymax':100, 'label':1}]
obboxes = [{'xmin':65, 'xmax':95, 'ymin':60, 'ymax':100, 'label':2}, 
           {'xmin':0, 'xmax':100, 'ymin':80, 'ymax':100, 'label':4}, 
           {'xmin':0, 'xmax':50, 'ymin':40, 'ymax':60, 'label': 3}]
rels = [[0,1,1], [1,0,2], [0,2,2], [0,2,3], [0,1,3]]
imageMeta = {'humans':hbboxes, 'objects':obboxes, 'rels':rels}


objpreds = np.zeros([6,6])
objpreds[0,1] = 1 # in gt
objpreds[1,1] = 1 # in gt

objpreds[2,4] = 1 # in gt
objpreds[3,2] = 1 # in gt
objpreds[4,5] = 1 # not in gt
objpreds[5,3] = 1 # in gt

bboxes = np.zeros([6,6*4])
bboxes[0,4:8]   = [1, 1, 18, 95]     # > max_overlap
bboxes[1,4:8]   = [50, 1, 80, 60]    # > min_overlap

bboxes[2,16:20] = [2, 5, 95, 82]     # < min_overlap + h=0: -3/na
bboxes[3,8:12]  = [60, 60, 100, 100] # > max_overlap + h=1: na/-1
bboxes[4,20:24] = [25, 25, 75, 75]   # no overlap: -2/-2
bboxes[5,12:16] = [0, 40, 51, 61]    # > max_overlap + h=0: 1/na
bboxes = bboxes / cfg.rpn_stride
#bboxes = bboxes.astype(np.int)

valid_human_boxes, valid_object_boxes, Xi, valid_labels, all_type = groundtruths.hoi_ground_truths(objpreds, bboxes, imageMeta, imageDims, cfg)