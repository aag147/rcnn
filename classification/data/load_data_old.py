# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:30:24 2018

@author: aag14
"""
from config import basic_config
from config_helper import set_config
import utils
import os

class data:
    def __init__(self, newDir=True):
        self.newDir = newDir
        self.cfg = None
        self.labels = None
        self.trainMeta = None
        self.valMeta = None
        self.testMeta = None
        self.trainGTMeta = None
        self.testGTMeta = None
        
        self.load_data()
        
        
    def remove_data(self):
        to_path = self.cfg.move_path
        utils.removeData(to_path)
        
    def move_data(self):
        from_path = self.cfg.data_path
        to_path   = self.cfg.move_path
        if to_path is None:
            return
        
        if not os.path.exists(to_path):
#            self.remove_data()        
            print('Moving data...')
            utils.moveData(from_path, to_path)
        print('Data already moved...')
        self.cfg.data_path = to_path+'/'

    def load_data(self):
        cfg = basic_config(self.newDir)
        cfg = set_config(cfg)
        cfg.get_args()
        cfg.update_paths()
        
        trainMeta = utils.load_dict(cfg.data_path + 'train')
        testMeta = utils.load_dict(cfg.data_path + 'test')
        
        trainGTMeta = utils.load_dict(cfg.data_path + 'train_GT')
        testGTMeta = utils.load_dict(cfg.data_path + 'test_GT')
        labels = utils.load_dict(cfg.data_path + 'labels')
        
        if cfg.max_classes is not None:
            # Reduce data to include only max_classes number of different classes
            _, counts = utils.getLabelStats(trainGTMeta, labels)
            trainGTMeta, reduced_idxs = utils.reduceTrainData(trainGTMeta, counts, cfg.max_classes)
            testGTMeta = utils.reduceTestData(testGTMeta, reduced_idxs)
            trainMeta = utils.reduceTestData(trainMeta, reduced_idxs)
            testMeta = utils.reduceTestData(testMeta, reduced_idxs)
            labels = utils.idxs2labels(reduced_idxs, labels)
            
            
        cfg.nb_classes = len(labels)
        cfg.set_class_weights(labels, trainGTMeta)
        _, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
        self.cfg = cfg
        
        if cfg.move:
            self.move_data()
            
        print('Path:', cfg.my_results_path)
        
        self.labels = labels
        self.trainMeta = trainMeta
        self.valMeta = valMeta
        self.testMeta = testMeta
        self.trainGTMeta = trainGTMeta
        self.testGTMeta = testGTMeta
