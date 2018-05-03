# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:37:55 2018

@author: aag14
"""
import sys
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')

import utils, draw, metrics
from load_data import data
from generators import DataGenerator

import numpy as np
import copy as cp
from matplotlib import pyplot as plt


data = data(newDir = False)
cfg = data.cfg

if True:
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  
    
    
    y_hat = utils.load_obj(cfg.part_results_path + 'y_hat_test_l')
    cm = metrics.computeConfusionMatrixLabels(genTest.gt_label, y_hat)
    
    _, counts = utils.getLabelStats(data.trainGTMeta, data.labels)
    idxs = np.concatenate([np.array([0]), np.argsort(counts, axis=0)+1], axis=0)
    
    _, test_counts = utils.getLabelStats(data.testGTMeta, data.labels)
    test_idxs = np.concatenate([np.array([0]), np.argsort(test_counts, axis=0)+ 1], axis=0)
    cm_sorted = cm[idxs,:]
#    cm_reduced = cm_sorted[-100:-1,0:]
    plt.figure(1)
    draw.plot_confusion_matrix(cm, data.labels, normalize=True)
    
    plt.figure(2)
    plt.plot([x for x in range(cm.shape[0])], cm_sorted[:,0] / np.sum(cm_sorted, axis=1))
    plt.ylabel('ratio of no label instances')
    plt.xlabel('classes')

evals = np.loadtxt(cfg.part_results_path + 'evals_m.txt', delimiter=', ')
evals = evals[1:-1:2]
hist = np.loadtxt(cfg.part_results_path + 'history_m.txt', delimiter=', ')
f, spl = plt.subplots(1,2)
spl = spl.ravel()
spl[0].plot([x for x in range(60)], evals)
spl[0].set_title('Test F1')
spl[1].plot([x for x in range(60)], hist[:,1])
spl[1].set_title('Training loss')