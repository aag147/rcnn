# -*- coding: utf-8 -*-
"""
Created on Mon May  7 08:55:19 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')
sys.path.append('../layers/')

import numpy as np
import cv2 as cv

       
def preprocessImage(img, cfg):
    shape = img.shape
    if shape[0] < shape[1]:
        newHeight = cfg.mindim
        scale = float(newHeight) / shape[0]
        newWidth = int(shape[1] * scale)
        if newWidth > cfg.maxdim:
            newWidth = cfg.maxdim
            scale = newWidth / shape[1]
            newHeight = shape[0] * scale
    else:
        newWidth = cfg.mindim
        scale = float(newWidth) / shape[1]
        newHeight = int(shape[0] * scale)
        if newHeight > cfg.maxdim:
            newHeight = cfg.maxdim
            scale = newHeight / shape[0]
            newWidth = shape[1] * scale

    scaleWidth = float(newWidth) / img.shape[1]
    scaleHeight = float(newHeight) / img.shape[0]
    scales = [scaleHeight, scaleWidth]
    # Rescale
    img = cv.resize(img, (newWidth, newHeight)).astype(np.float32)
    # Normalize
    img = (img - np.min(img)) / np.max(img)
    # Transpose
    img = img.transpose(cfg.order_of_dims)
    return img, scales

       
def normalizeGTboxes(gtboxes, scale=[1,1], rpn_stride=1, shape=[1,1]):
    gtnormboxes = []
    for relID, bbox in enumerate(gtboxes):
        # get the GT box coordinates, and resize to account for image resizing
        xmin = (bbox['xmin'] * scale[0] / rpn_stride) / shape[0]
        xmax = (bbox['xmax'] * scale[0] / rpn_stride) / shape[0]
        ymin = (bbox['ymin'] * scale[1] / rpn_stride) / shape[1]
        ymax = (bbox['ymax'] * scale[1] / rpn_stride) / shape[1]
        gtnormboxes.append({'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax})    
    return gtnormboxes
