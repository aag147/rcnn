# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:26:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../fast_rcnn/')
sys.path.append('../cfgs/')
sys.path.append('../layers/')

from config import basic_config
import utils
from detection_generators import DataGenerator
import numpy as np


# Config
cfg = basic_config(False)
cfg.fast_rcnn_config()
cfg.dataset = 'COCO'
cfg.part_results_path = 'C:/Users/aag14/Documents/Skole/Speciale/results/'
cfg.part_data_path  = 'C:/Users/aag14/Documents/Skole/Speciale/data/'
cfg.weights_path  = 'C:/Users/aag14/Documents/Skole/Speciale/weights/'
cfg.update_paths()


cfg.nb_classes = 81

cfg.rpn_stride = 16

cfg.anchor_sizes = [128, 256, 512]
cfg.anchor_ratios = [[1, 1], [1, 2], [2, 1]]

cfg.rpn_min_overlap = 0.1
cfg.rpn_max_overlap = 0.5

# Data
valGTMeta = utils.load_dict(cfg.data_path + 'val_GT')
class_mapping = utils.load_dict(cfg.data_path + 'class_mapping')

# Create batch generators
genVal = DataGenerator(imagesMeta = valGTMeta, cfg=cfg, data_type='val').begin()


# test
deltas_to_roi(rpn_layer, regr_layer, cfg)
reduce_rois(true_labels, cfg)
detection_ground_truths(rois, imageMeta, imageDims, cfg, class_mapping)