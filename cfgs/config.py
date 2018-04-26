# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""
import method_configs as mcfg
import utils

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
      if len(self.my_results_path) > 0 or not self.newDir:
          return
      
      for fid in range(100):
        path = self.part_results_path + self.dataset + "/" + self.modelnamekey + '%d/' % fid
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path + 'weights/')
            break
      self.my_results_path = path
      self.my_weights_path = path + 'weights/'
      
      
   def update_paths(self):
       self.get_data_path()
       self.get_results_paths()
      
   def __init__(self, newDir = True):
       self.newDir = newDir
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
       self.testdata = 'genTest'
       
       #model
       self.task = None
       self.pretrained_weights = True
       self.my_weights = None
       
       #model compile
       self.optimizer = 'sgd'
       self.wp = 1
       
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
       
       self.shape = (self.ydim, self.xdim)
       self.order_of_dims = [2,0,1]
       self.par_order_of_dims = [0,1,2,3]
       self.winShape = (64, 64)
       
   def fast_rcnn_config(self):
       self.xdim=224
       self.ydim=224
       self.cdim=3
       
       self.pool_size = 7
       
       self.shape = (self.ydim, self.xdim)
       self.order_of_dims = [0,1,2]
       self.par_order_of_dims = [0,2,3,1]
       self.winShape = (64, 64)
       
   def get_args(self):
       try:
          argv = sys.argv[1:]
          opts, args = getopt.getopt(argv,"m:c:x:d:w:v:t:b:")
       except getopt.GetoptError:
          print('.py -m <my_model> -c <my_method> -x <max_classes> -d <dataset>')
          sys.exit(2)
     
       for opt, arg in opts:
          print(opt, arg)
          if opt == '-v':
             path = self.part_results_path + arg
             self.my_results_path = path
             self.my_weights_path = path + 'weights/'
          if opt == '-m':
             self.my_weights = arg
          if opt == '-c':
             assert hasattr(mcfg, arg), 'method cfg needs to exist'
             self = getattr(mcfg, arg)(self)
          if opt == '-x':
              assert arg.isdigit(), 'max_classes must be int'
              self.max_classes = int(arg)
          if opt == '-b':
              assert arg.isdigit(), 'nb_batches must be int'
              self.train_cfg.nb_batches = int(arg)
              self.val_cfg.nb_batches = int(arg)
          if opt == '-d':
              self.dataset = arg
          if opt == '-w':
              assert (arg.isdigit() or arg=='-1'), 'weight must be int'
              self.wp = int(arg)
          if opt == '-t':
              self.testdata = arg
              
   def set_class_weights(self, labels, imagesMeta):
       if self.wp >= 0: 
           return
       print('Using class-specific weights!')
       stats, counts = utils.getLabelStats(imagesMeta, labels)
       print(labels)
       print(counts)
       print(stats)
       p = counts / sum(counts)
       wp = 1 / p
       self.wp = wp
