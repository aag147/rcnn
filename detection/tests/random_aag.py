# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 10:06:36 2018

@author: aag14
"""
import filters_helper as helper

img_ids, category_ids = helper.getCOCOIDs(data.testGTMeta, class_mapping)
path = cfg.part_data_path + 'COCO/img5_ids'
utils.save_dict(img_ids, path)
path = cfg.part_data_path + 'COCO/category5_ids'
utils.save_dict(category_ids, path)


#cocoform = helper.bboxes2COCOformat(boxes_nms, imageMeta, class_mapping, imageDims['scale'], cfg.rpn_stride)
#path = cfg.part_data_path + 'COCO/results'
#utils.save_dict(cocoform, path)