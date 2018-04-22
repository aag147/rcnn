# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""
import method_configs as mcfg

import os
import sys, getopt

class basic_config:
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
       self.setBasicValues()
       self.rcnn_config() #standard
       
   def setBasicValues(self):
       #paths
       self.part_results_path = ''
       self.part_data_path  = ''
       self.weights_path = ''
       self.my_results_path = ''
       self.my_weights_path = ''
       
       #basics
       self.dataset = 'HICO'
       self.inputs  = None
       self.max_classes = None
       
       #generator
       self.train_cfg = self.gen_config()
       self.test_cfg = self.gen_config()
       self.val_cfg = self.gen_config()
       self.xdim= None
       self.ydim= None
       self.cdim= None
       self.minIoU = 0.5
       
       #model
       self.task = None
       self.pretrained_weights = True
       self.my_weights = None
       
       #model compile
       self.optimizer = 'sgd'
       self.wp = None
       
       #model callbacks
       self.patience = None
       self.modelnamekey = ''
       self.epoch_splits = None
       self.init_lr = None
       self.include_eval = False
       self.checkpoint_interval = 10
       
       # model training
       self.epoch_begin = None
       self.epoch_end = None
       
   def rcnn_config(self):
       self.xdim=227
       self.ydim=227
       self.cdim=3
       
   def fast_rcnn_config(self):
       self.xdim=227
       self.ydim=227
       self.cdim=3
       
   def get_args(self):
       try:
          argv = sys.argv[1:]
          opts, args = getopt.getopt(argv,"m:c:x:d:")
       except getopt.GetoptError:
          print('.py -m <my_model> -c <my_method> -x <max_classes> -d <dataset>')
          sys.exit(2)
     
       for opt, arg in opts:
          print(opt, arg)
          if opt == '-m':
             self.my_weights = arg
          if opt == '-c':
             assert hasattr(mcfg, arg), 'method cfg needs to exist'
             self = getattr(mcfg, arg)(self)
          if opt == '-x':
              assert isinstance(arg, int), 'max_classes must be int'
              self.max_classes = int(arg)
          if opt == '-d':
              self.dataset = arg
          if opt == '-w':
              assert isinstance(arg, int), 'weight must be int'
              self.wp = arg
