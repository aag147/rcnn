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
import rpn_test
import os


if True:
    # meta data
    print('Loading data...')
    data = extract_data.object_data(False)
    cfg = data.cfg
    
    cfg.my_save_path = cfg.data_path +'anchors/'
    if not os.path.exists(cfg.my_save_path):
        os.makedirs(cfg.my_save_path)

    # data
    trainMeta = data.trainGTMeta
    valMeta = data.valGTMeta
    testMeta = data.testGTMeta

# Save RPN input data
rpn_test.saveInputData(testMeta, 'test', cfg)
#rpn_test.saveInputData(valMeta, 'val', cfg)
#rpn_test.saveInputData(trainMeta, 'train', cfg)
