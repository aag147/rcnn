# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:48:23 2018

@author: aag14
"""


import sys 
sys.path.append('../../../')
sys.path.append('../../classification/data/')
sys.path.append('../../classification/models/')
sys.path.append('../../shared/')

import utils
from model_trainer import model_trainer
from load_data import data
from generators import DataGenerator

import numpy as np
import metrics
import image



if False:
    # Load data
    print('Loading data...')
    data = data(method='normal')
    cfg = data.cfg


    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  

    gt_label, _, _ = image.getYData(genTest.dataID, genTest.imagesMeta, genTest.GTMeta, genTest.cfg)
    
path = 'C:\\Users\\aag14/Documents/Skole/Speciale/results/HICO/hoi80/yhat1'
yhat = utils.load_obj(path)

evalHOI = metrics.EvalResults(None, genTest, yhat=yhat, y=gt_label[0])
print(evalHOI.mAP, evalHOI.F1)