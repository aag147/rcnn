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

path = cfg.my_results_path + 'history.txt'
hist =  np.loadtxt(path, delimiter=', ')

import draw
draw.plotRPNLosses(hist)