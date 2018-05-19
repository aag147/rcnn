# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:30:24 2018

@author: aag14
"""
from config import basic_config
from config_helper import set_config
import utils
import os
import numpy as np

class object_data:
    def __init__(self, newDir=True):
        self.newDir = newDir
        self.cfg = None
        self.labels = None
        self.trainGTMeta = None
        self.testGTMeta = None
        self.valGTMeta = None
        
        
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
        cfg.fast_rcnn_config()
        cfg = set_config(cfg)
        cfg.get_args()
        cfg.update_paths()
        
        trainGTMeta = utils.load_dict(cfg.data_path + 'train_GT')
        valGTMeta = utils.load_dict(cfg.data_path + 'val_GT')
#        testGTMeta = utils.load_dict(cfg.data_path + 'test_GT')
        class_mapping = utils.load_dict(cfg.data_path + 'class_mapping')
        
        if cfg.max_classes is not None:
            # Reduce data to include only max_classes number of different classes
#            _, counts = utils.getLabelStats(trainGTMeta, class_mapping)
            reduced_objs = self.getReduxIdxs(class_mapping)
            trainGTMeta = self.reduceData(trainGTMeta, reduced_objs)
            class_mapping = self.reduceMapping(reduced_objs)
#            testGTMeta = utils.reduceTestData(testGTMeta, reduced_idxs)
#            trainMeta = utils.reduceTestData(trainMeta, reduced_idxs)
#            testMeta = utils.reduceTestData(testMeta, reduced_idxs)
#            valGTMeta = utils.reduceTestData(valGTMeta, reduced_idxs)
#            class_mapping = utils.idxs2labels(reduced_idxs, class_mapping)
            
            
        cfg.nb_object_classes = len(class_mapping)
#        cfg.set_class_weights(class_mapping, trainGTMeta)
#        _, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
        self.cfg = cfg
        
        if cfg.move:
            self.move_data()
            
        print('Path:', cfg.my_results_path)
        
        self.class_mapping = class_mapping
        self.valGTMeta = valGTMeta
#        self.trainMeta = trainMeta
#        self.valMeta = valMeta
#        self.testMeta = testMeta
        self.trainGTMeta = trainGTMeta
#        self.testGTMeta = testGTMeta
        
    def getReduxIdxs(self, class_mapping):
        objs = utils.getPascalObjects()
        reduced_idxs = []
        for label, idx in class_mapping.items():
            if label in objs:
                reduced_idxs.append(idx)
        return objs
    
    def reduceData(self, imagesMeta, reduced_objs):
        reduced_imagesMeta = {}
        i = 0
        j = 0
        k = 0
        for imageID, imageMeta in imagesMeta.items():
            new_rels = []
            new_prs = []
            for idx, obj in enumerate(imageMeta['objects']):
                if obj['label'] == 'person':
                    new_prs.append(obj)
                elif obj['label'] in reduced_objs:
                    new_rels.append(obj)
                    
            if len(new_rels) > 0:
                j += 1
                reduced_imagesMeta[imageID] = {'imageName':imageMeta['imageName'], 'objects':new_prs+new_rels}
        print(i,j,k)
        return reduced_imagesMeta   

    def reduceMapping(self, reduced_objs):
        new_class_mapping = {'bg': 0, 'person': 1}
        for obj in reduced_objs:
            if obj == 'person':
                continue
            new_class_mapping[obj] = len(new_class_mapping)
        return new_class_mapping
    
    def getBareBonesStats(self, class_mapping):
        stats = {}
        for label, idx in class_mapping.items():
            if label not in stats:
                stats[label] = 0
        return stats
    
    
    def getLabelStats(self, imagesMeta, class_mapping):
        stats = self.getBareBonesStats(class_mapping)
        stats['total'] = 0
        stats['nb_samples'] = 0
        stats['nb_images'] = 0
        for imageID, imageMeta in imagesMeta.items():
            stats['nb_images'] += 1
            for rel in imageMeta['objects']:
                stats['nb_samples'] += 1
                stats['total'] += 1
                stats[rel['label']] += 1
                    
        return stats
