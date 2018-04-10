# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""

class config:
   def get_data_path(self):
       return self.part_data_path + self.dataset
   
   def __init__(self, nb_classes, dataset):
       #basics
       self.dataset = dataset
       self.results_path = ''
       self.weights_path = ''
       self.part_data_path  = ''
       self.nb_classes = nb_classes
       
       #generator
       self.train_type = 'itr'
       self.val_type = 'itr'
       self.test_type = 'itr'
       self.xdim=227
       self.ydim=227
       self.cdim=3
       
       #model
       self.task = 'multi-label'
       self.pretrained_weights = True
       
       #model compile
       self.optimizer = 'sgd'
       self.wp = 20
       
       #model callbacks
       self.patience = 0
       self.modelnamekey = ''
       self.epoch_splits = [5]
       self.init_lr = 0.001
       self.include_eval = False
       
       # model training
       self.epoch_begin = 0
       self.epoch_end = 5