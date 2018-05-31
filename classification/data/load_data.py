# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:30:24 2018

@author: aag14
"""
from config import basic_config
from config_helper import set_config
import method_configs as mcfg


import utils
import os

class data:
    def __init__(self, newDir=True, method='normal'):
        self.newDir = newDir
        self.method = method
        
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
        
        to_path += '/'+self.cfg.dataset
        
        if not os.path.exists(to_path):
#            self.remove_data()        
            print('Moving data...')
            utils.moveData(from_path, to_path)
        print('Data already moved...')
        self.cfg.data_path = to_path+'/'

    def load_data(self):
        cfg = basic_config(self.newDir)
        
        if self.method == 'faster':
            cfg.faster_rcnn_config()
        elif self.method == 'fast':
            cfg.fast_rcnn_config()
        elif self.method == 'normal':
            cfg.rcnn_config()

        cfg = mcfg.rcnn_hoi_classes(cfg)
        cfg = set_config(cfg)
        cfg.get_args()
        cfg.dataset = 'HICO'
        cfg.update_paths()
        
        trainMeta = utils.load_dict(cfg.data_path + 'train')
        testMeta = utils.load_dict(cfg.data_path + 'test')
        
        trainGTMeta = utils.load_dict(cfg.data_path + 'train_GT')
        testGTMeta = utils.load_dict(cfg.data_path + 'test_GT')
        labels = utils.load_dict(cfg.data_path + 'labels')
        class_mapping = utils.load_dict(cfg.data_path + 'class_mapping')
        
        if cfg.max_classes is not None:
            # Reduce data to include only max_classes number of different classes
            _, counts = utils.getLabelStats(trainGTMeta, labels)
            reduced_idxs = utils.getReducedIdxs(counts, cfg.max_classes, labels)
            trainGTMeta = utils.reduceData(trainGTMeta, reduced_idxs)
            testGTMeta = utils.reduceData(testGTMeta, reduced_idxs)
            trainMeta = utils.reduceData(trainMeta, reduced_idxs)
            testMeta = utils.reduceData(testMeta, reduced_idxs)
            labels = utils.idxs2labels(reduced_idxs, labels)
            
            
        cfg.nb_classes = len(labels)
        cfg.set_class_weights(labels, trainGTMeta)
        _, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
        self.cfg = cfg
        
        if cfg.move:
            self.move_data()
        
        print('Data:', cfg.data_path)
        print('Path:', cfg.my_results_path)
        
        self.labels = labels
        self.class_mapping = class_mapping
        self.trainMeta = trainMeta
        self.valMeta = valMeta
        self.testMeta = testMeta
        self.trainGTMeta = trainGTMeta
        self.testGTMeta = testGTMeta
