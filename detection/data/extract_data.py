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
import numpy as np

class object_data:
    def __init__(self, newDir=True, method='faster'):
        self.newDir = newDir
        self.method = method
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
        to_path += '/'+self.cfg.dataset
        
        if not os.path.exists(to_path):
#            self.remove_data()        
            print('Moving data...')
            utils.moveData(from_path, to_path)
        print('Data already moved...')
        self.cfg.data_path = to_path+'/'
        
        self.cfg.my_save_path = self.cfg.move_path + '/results/' + self.cfg.new_results_dir

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
        cfg.update_paths()
        
        trainGTMeta = utils.load_dict(cfg.data_path + 'train_objs')
#        valGTMeta = utils.load_dict(cfg.data_path + 'val_objs')
        testGTMeta = utils.load_dict(cfg.data_path + 'test_objs')
        class_mapping = utils.load_dict(cfg.data_path + 'class_mapping')
        hoi_labels = None
        reduced_hoi_map = None
        
        if os.path.exists(cfg.data_path + 'labels.JSON'):
            hoi_labels = utils.load_dict(cfg.data_path + 'labels')
            
            
        if cfg.max_classes is not None:
            # Reduce data to include only max_classes number of different classes
#            _, counts = utils.getLabelStats(trainGTMeta, class_mapping)
            reduced_objs = self.getReduxIdxs(class_mapping, cfg)
            
            class_mapping = self.reduceMapping(reduced_objs)
            if hoi_labels is not None:
                hoi_labels, reduced_hoi_map = self.reduceHoILabels(hoi_labels, reduced_objs)
            
            trainGTMeta = self.reduceData(trainGTMeta, reduced_objs, reduced_hoi_map)
            testGTMeta = self.reduceData(testGTMeta, reduced_objs, reduced_hoi_map)
#            valGTMeta = self.reduceData(valGTMeta, reduced_objs)
#            trainMeta = utils.reduceTestData(trainMeta, reduced_idxs)
#            testMeta = utils.reduceTestData(testMeta, reduced_idxs)
#            class_mapping = utils.idxs2labels(reduced_idxs, class_mapping)
            
            
        cfg.nb_object_classes = len(class_mapping)
        cfg.nb_classes = len(hoi_labels) if hoi_labels is not None else 0
        cfg.nb_hoi_classes = cfg.nb_classes
#        cfg.set_class_weights(class_mapping, trainGTMeta)
#        _, valMeta = utils.splitData(list(trainMeta.keys()), trainMeta)
        
        self.cfg = cfg
        
        if cfg.move:
            self.move_data()
            
        print('Path:', cfg.my_results_path)
        
        self.class_mapping = class_mapping
        self.testGTMeta = testGTMeta
        self.trainGTMeta = trainGTMeta
        
        self.hoi_labels = hoi_labels
#        self.trainMeta = trainMeta
#        self.valMeta = valMeta
#        self.testMeta = testMeta
#        self.testGTMeta = testGTMeta
        
    def getReduxIdxs(self, class_mapping, cfg):
        objs = utils.getPascalObjects(cfg.max_classes)
#        reduced_idxs = []
#        for label, idx in class_mapping.items():
#            if label in objs:
#                reduced_idxs.append(idx)
        return objs
    
    def reduceData(self, imagesMeta, reduced_objs, reduced_hoi_map=None):
        reduced_imagesMeta = {}
        for imageID, imageMeta in imagesMeta.items():
            new_objs = []
            new_rels = []
            objsidxs = {}
            hasObjs = False
            for idx, obj in enumerate(imageMeta['objects']):
                if obj['label'] not in reduced_objs:
                    continue
                if obj['label'] != 'person':
                    hasObjs = True
                new_objs.append(obj)
                objsidxs[idx] = len(new_objs) - 1
                    
            if hasObjs:
                if 'rels' in imageMeta:
                    for idx, rel in enumerate(imageMeta['rels']):    
                        if rel[0] in objsidxs and rel[1] in objsidxs:
                            new_rels.append([objsidxs[rel[0]], objsidxs[rel[1]], reduced_hoi_map[rel[2]]])
                    new_rels = np.array(new_rels)

                reduced_imagesMeta[imageID] = {'imageName':imageMeta['imageName'], 'objects':new_objs, 'rels':new_rels}
        return reduced_imagesMeta   

    def reduceMapping(self, reduced_objs):
        new_class_mapping = {'bg': 0, 'person': 1}
        for obj in reduced_objs:
            if obj == 'person':
                continue
            new_class_mapping[obj] = len(new_class_mapping)
        return new_class_mapping
    
    def reduceHoILabels(self, hoi_labels, reduced_objs):
        new_labels = []
        reduced_map = {}
        for idx, label in enumerate(hoi_labels):
            if label['obj'] in reduced_objs:
                new_labels.append(label)
                reduced_map[idx] = len(reduced_map)
        return new_labels, reduced_map
    
    def getBareBonesStats(self, class_mapping):
        stats = {}
        for label, idx in class_mapping.items():
            if label not in stats:
                stats[label] = 0
        return stats
    
    
    def getLabelStats(self, dataset='train'):
        if dataset == 'test':
            imagesMeta = self.testGTMeta
        else:
            imagesMeta = self.trainGTMeta
        stats = self.getBareBonesStats(self.class_mapping)
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
