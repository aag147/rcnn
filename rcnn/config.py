# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""
import os

class config:
   class gen_config:
       def __init__(self):
           self.type = 'itr'
           self.batch_size = 32
           self.nb_batches = None
           self.images_per_batch = 16
           self.shuffle = False
    
   def get_data_path(self):
       self.data_path = self.part_data_path + self.dataset + "/"
   
   def get_results_paths(self):
      for fid in range(100):
        path = self.part_results_path + self.dataset + "/" + self.modelnamekey + '%d/' % fid
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path + 'weights/')
            break
      self.my_results_path = path
      self.my_weights_path = path + 'weights/'
      
   def __init__(self):
       #basics
       self.dataset = 'TU_PPMI'
       self.inputs  = [1,1,1]
       self.max_classes = None
       self.part_results_path = ''
       self.part_data_path  = ''
       self.weights_path = ''
       self.my_results_path = ''
       self.my_weights_path = ''
       
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
       self.my_weights = None
       
       #model compile
       self.optimizer = 'sgd'
       self.wp = 20
       
       #model callbacks
       self.patience = 0
       self.modelnamekey = ''
       self.epoch_splits = [5]
       self.init_lr = 0.001
       self.include_eval = False
       self.checkpoint_interval = 10
       
       # model training
       self.epoch_begin = 0
       self.epoch_end = 5