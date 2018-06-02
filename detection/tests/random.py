# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 10:06:36 2018

@author: aag14
"""
import filters_helper as helper



cocoform = helper.bboxes2COCOformat(boxes_nms, imageMeta, class_mapping, imageDims['scale'], cfg.rpn_stride)
path = cfg.part_data_path + 'COCO/results'
utils.save_dict(cocoform, path)