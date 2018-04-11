# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""
import utils

class config:
   class gen_config:
       def __init__(self):
           self.type = 'itr'
           self.batch_size = 32
           self.shuffle = False
    
   def get_data_path(self):
       return self.part_data_path + self.dataset
   
   def get_nb_classes(self):
       return utils.getUniqueClasses(self)
   
   def __init__(self):
       #basics
       self.dataset = 'TU_PPMI'
       self.results_path = ''
       self.weights_path = ''
       self.part_data_path  = ''
       
       #generator
       self.train_cfg = self.gen_config()
       self.test_cfg = self.gen_config()
       self.val_cfg = self.gen_config()
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