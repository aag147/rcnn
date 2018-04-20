# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:30:24 2018

@author: aag14
"""
from config import basic_config
from config_helper import set_config
import utils

class data:
    def __init__(self):
        self.cfg = None
        self.labels = None
        self.trainMeta = None
        self.valMeta = None
        self.testMeta = None
        self.trainGTMeta = None
        self.testGTMeta = None
        
        self.load_data()

    def load_data(self):
        cfg = basic_config()
        cfg.get_args()
        cfg = set_config(cfg)
        
        trainMeta = utils.load_dict(cfg.data_path + 'train')
        testMeta = utils.load_dict(cfg.data_path + 'test')
        
        trainGTMeta = utils.load_dict(cfg.data_path + 'train_GT')
        testGTMeta = utils.load_dict(cfg.data_path + 'test_GT')
        labels = utils.load_dict(cfg.data_path + 'labels')
        print('Path:', cfg.my_results_path)
        
        if cfg.max_classes is not None:
            # Reduce data to include only max_classes number of different classes
            _, counts = utils.getLabelStats(trainGTMeta, labels)
            trainGTMeta, reduced_idxs = utils.reduceTrainData(trainGTMeta, counts, cfg.max_classes)
            testGTMeta = utils.reduceTestData(testGTMeta, reduced_idxs)
            trainMeta = utils.reduceTestData(trainMeta, reduced_idxs)
            testMeta = utils.reduceTestData(testMeta, reduced_idxs)
            labels = utils.idxs2labels(reduced_idxs, labels)
            
            
        cfg.nb_classes = len(labels)
        trainMeta, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
        self.cfg = cfg
        self.labels = labels
        self.trainMeta = trainMeta
        self.valMeta = valMeta
        self.testMeta = testMeta
        self.trainGTMeta = trainGTMeta
        self.testGTMeta = testGTMeta