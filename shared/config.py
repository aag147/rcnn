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
      if len(self.my_results_dir) > 0:
          path = self.part_results_path + self.dataset + "/" + self.my_results_dir + '/'
      else:
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
       self.my_results_dir = ''
       self.move_path = None
       self.move = 0
       
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
       self.include_eval = True
       self.include_validation = False
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
       self.mindim = 400
       self.maxdim = 600
       self.xdim = 224
       self.ydim = 224
       self.cdim  = 3
       
       self.pool_size = 7
       
       self.order_of_dims = [0,1,2]
       self.par_order_of_dims = [0,2,3,1]
       self.winShape = (64, 64)
       
   def get_args(self):
       try:
          argv = sys.argv[1:]
          opts, args = getopt.getopt(argv,"ab:c:d:e:f:g:hi:l:m:n:o:r:s:tuw:x:")
       except getopt.GetoptError:
          print('.py -m <my_model> -c <my_method> -x <max_classes> -d <dataset>')
          sys.exit(2)
     
#    augment, backbone, cfg_method, dataset, epoch_split, final_epoch, generator_type, input_roi_dir, learning_rate, model, nb_batches, optimizer, results_dir, start_epoch, transfor data, uniform_sampling, weighing, ma(x)_classes
       for opt, arg in opts:
          print(opt, arg)
          if opt == '-a':
             # augmentation
             self.flip_image = True
          if opt == '-b':
             # backbone
             self.backbone = arg
          if opt == '-c':
             # cfg method
             assert hasattr(mcfg, arg), 'method cfg needs to exist'
             self = getattr(mcfg, arg)(self)
          if opt == '-d':
              # dataset
              self.dataset = arg
          if opt == '-e':
              # epoch learning rate split
              assert arg.isdigit(), 'epoch learning split must be int'
              self.epoch_splits = [int(arg)]
          if opt == '-f':
              # final epoch
              assert arg.isdigit(), 'final epoch must be int'
              self.epoch_end = int(arg)
          if opt == '-g':
              # generator iterator type
              self.train_cfg.type = arg
          if opt == '-h':
              # fine-tune with shared CNN
              self.use_shared_cnn = True
          if opt == '-i':
              # roi input directory for detection
              self.my_detections_dir = arg
          if opt == '-l':
              # initial learning rate
              self.init_lr = float(arg)
          if opt == '-m':
              # loadable model/weights
              self.my_weights = arg
          if opt == '-n':
              # number of batches per epoch
              assert arg.isdigit(), 'nb_batches must be int'
              self.train_cfg.nb_batches = int(arg)
          if opt == '-o':
              # optimizer
              self.optimizer = arg
          if opt == '-r':
              # use results directory from previous model
              self.my_results_dir = arg
          if opt == '-s':
              # start epoch
              assert arg.isdigit(), 'start epoch must be int'
              self.epoch_begin = int(arg)
          if opt == '-t':
              # transfer data to scratch
              self.move = True
          if opt == '-u':
              # use uniform rpn proposal sampling
              self.rpn_uniform_sampling = True
          if opt == '-w':
              # weighing in loss
              assert (arg.isdigit() or arg=='-1'), 'weight must be int'
              self.wp = int(arg)
          if opt == '-x':
              # max classes
              assert arg.isdigit(), 'max_classes must be int'
              self.max_classes = int(arg)
              
   def set_class_weights(self, labels, imagesMeta):
       if self.wp >= 0: 
           return
       print('Using class-specific weights!')
       stats, counts = utils.getLabelStats(imagesMeta, labels)
       p = counts / sum(counts)
       wp = 1 / p
       self.wp = wp
