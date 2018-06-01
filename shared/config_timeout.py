# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""
import method_configs as mcfg
import utils

import os
import sys, getopt
import numpy as np

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
      if len(self.my_results_dir) > 0 and not self.use_shared_cnn:
          path = self.part_results_path + self.my_results_dir + '/'
      else:
          for fid in range(100):
            path = self.part_results_path + self.dataset + "/" + self.modelnamekey + '%d/' % fid
            if not os.path.exists(path):
                os.mkdir(path)
                os.mkdir(path + 'weights/')
                break
            
      self.my_results_path = path
      self.my_weights_path = path + 'weights/'
      
      if self.my_weights is not None:
          self.my_shared_weights = self.my_weights_path + self.my_weights
      
      if self.use_shared_cnn:
          self.my_shared_weights = self.part_results_path + self.my_results_dir + '/weights/' + self.my_weights
      
   def get_detections_path(self):
       if len(self.my_detections_dir) == 0:
          return
       
       self.my_detections_path = self.part_results_path + self.dataset + "/" + self.my_detections_dir + '/detections/'
      
      
   def update_paths(self):
       self.get_data_path()
       self.get_results_paths()
       self.get_detections_path()
      
   def __init__(self, newDir = True):
       self.newDir = newDir
       self.setBasicValues()
       
   def setBasicValues(self):
       #paths
       self.part_results_path = ''
       self.part_data_path  = ''
       self.weights_path = ''
       self.my_results_path = ''
       self.my_weights_path = ''
       self.my_results_dir = ''
       self.my_detections_dir = ''
       self.my_detections_path = ''
       self.move_path = None
       self.move = False
       self.use_shared_cnn = False
       
       #basics
       self.dataset = 'HICO'
       self.inputs  = None
       self.max_classes = None
       self.backbone = None
       
       #generator
       self.train_cfg = self.gen_config()
       self.test_cfg = self.gen_config()
       self.val_cfg = self.gen_config()
       self.xdim= None
       self.ydim= None
       self.cdim= None
       self.minIoU = 0.5
       self.testdata = 'genTest'
       
       self.img_channel_mean = np.array([[[102.9801, 115.9465, 122.7717]]]) #BGR
       self.img_scaling_factor = 1.0
       
       self.order_of_dims = [0,1,2]
       self.par_order_of_dims = [0,2,3,1]
       self.winShape = (64, 64)
       
       #model
       self.task = None
       self.pretrained_weights = True
       self.my_weights = None
       
       #model compile
       self.optimizer = 'sgd'
       self.wp = 1
       
       #model callbacks
       self.patience = 100
       self.modelnamekey = ''
       self.epoch_splits = [100]
       self.init_lr = None
       self.include_eval = True
       self.include_validation = False
       self.checkpoint_interval = 10
       
       # model training
       self.epoch_begin = None
       self.epoch_end = None
       
       
       #fast data preprocesses
       self.use_channel_mean = False
       self.rpn_uniform_sampling = False
       self.flip_image = False
       self.pool_size = None
       
   def rcnn_config(self):
       self.xdim=227
       self.ydim=227
       self.cdim=3
              
       # Basic stuff
       self.pool_size = 3
       self.init_lr = 0.0001
       self.epoch_begin = 0
       self.epoch_end = 60
       self.epoch_splits = [40]
       self.optimizer = 'sgd'
       self.backbone = 'alex'
       
       self.shape = (self.ydim, self.xdim)       

       
   def fast_rcnn_config(self):
       self.mindim = 600
       self.maxdim = 1000
       self.cdim  = 3
       
       
       # Basic stuff
       self.pool_size = 3
       self.init_lr = 0.00001
       self.epoch_begin = 0
       self.epoch_end = 60
       self.optimizer = 'adam'
       self.backbone = 'alex'
       
       self.train_cfg.batch_size = 1
       self.val_cfg.batch_size = 1
       self.test_cfg.batch_size = 1      
       
   def faster_rcnn_config(self):
       self.mindim = 600
       self.maxdim = 1000
       self.cdim  = 3
       
       # Basic stuff
       self.pool_size = 3
       self.init_lr = 0.00001
       self.nb_batches = 1000
       self.epoch_begin = 0
       self.epoch_end = 60
       self.optimizer = 'adam'
       
       self.order_of_dims = [0,1,2]
       self.par_order_of_dims = [0,2,3,1]
       self.winShape = (64, 64)
       
       self.train_cfg.batch_size = 1
       self.val_cfg.batch_size = 1
       self.test_cfg.batch_size = 1
       
       self.rpn_regr_std = 4.0
       self.det_regr_std = [8.0, 8.0, 4.0, 4.0]
       
       #rpn filters
       self.nb_shared_layers = 17
       self.rpn_stride = 16
       self.nb_rpn_proposals = 128
        
       self.anchor_sizes = [64, 128, 256, 512]
       self.anchor_ratios = [[1, 1], [1, 2], [2, 1]]
        
       self.rpn_min_overlap = 0.3
       self.rpn_max_overlap = 0.7
       
       self.rpn_nms_max_boxes=300
       self.rpn_nms_overlap_thresh=0.7

        
       # detection filters
       self.detection_max_overlap = 0.5
       self.detection_min_overlap = 0.0
       self.nb_detection_rois = 64
       
       self.det_nms_max_boxes=300
       self.det_nms_overlap_thresh=0.90
        
       # hoi filters
       self.hoi_max_overlap = 0.5
       self.hoi_min_overlap = 0.1
       self.hoi_nms_overlap_thresh=0.5
       
       self.hoi_pos_share  = 4
       self.hoi_neg1_share = 12
       self.hoi_neg2_share = 16
       
       self.nb_hoi_rois = 32
        
       # model
       self.nb_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
#       self.nb_object_classes = 81
       self.nb_hoi_classes = 600
       
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
       
       
       
### OLD ARG PARSER
#        for opt, arg in opts:
#          print(opt, arg)
#          if opt == '-v':
##             path = self.part_results_path + arg
#             self.my_results_dir = arg
##             self.my_results_path = path
##             self.my_weights_path = path + 'weights/'
#          if opt == '-m':
#             self.my_weights = arg
#          if opt == '-c':
#             assert hasattr(mcfg, arg), 'method cfg needs to exist'
#             self = getattr(mcfg, arg)(self)
#          if opt == '-x':
#              assert arg.isdigit(), 'max_classes must be int'
#              self.max_classes = int(arg)
#          if opt == '-b':
#              assert arg.isdigit(), 'nb_batches must be int'
#              self.train_cfg.nb_batches = int(arg)
##              self.val_cfg.nb_batches = int(arg)
#          if opt == '-d':
#              self.dataset = arg
#          if opt == '-w':
#              assert (arg.isdigit() or arg=='-1'), 'weight must be int'
#              self.wp = int(arg)
#          if opt == '-t':
#              self.testdata = arg
#          if opt == '-n':
#              assert arg.isdigit(), 'move must be int'
#              self.move = int(arg)
#          if opt == '-s':
#              assert arg.isdigit(), 'final epoch must be int'
#              self.epoch_begin = int(arg)
#          if opt == '-f':
#              assert arg.isdigit(), 'final epoch must be int'
#              self.epoch_end = int(arg)
#          if opt == '-h':
#              assert arg.isdigit(), 'epoch learning split must be int'
#              self.epoch_splits = [int(arg)]
#          if opt == '-o':
#              self.optimizer = arg
#          if opt == '-l':
#              self.init_lr = float(arg)
#          if opt == '-g':
#              self.train_cfg.type = arg
#          if opt == '-u':
#              assert arg.isdigit(), 'uniform flag must be int'
#              if int(arg)==0:
#                  self.rpn_uniform_sampling = False
#          if opt == '-r':
#              if int(arg)==0:
#                  self.use_channel_mean = False
#          if opt == '-p':
#              self.my_detections_dir = arg
#          if opt == '-a':
#              self.flip_image = True
