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
           self.batch_size = 64
           self.nb_batches = None
           self.images_per_batch = 16
           self.shuffle = False
    
   def get_data_path(self):
       self.local_base_path = self.base_path
       self.data_path = self.part_data_path + self.dataset + "/"
       self.results_path = self.part_results_path + self.dataset + "/"
   
   def get_results_paths(self):
      if not self.newDir:
          print("   No directory (test)...")
          self.my_actual_results_dir = ''
          return
      elif len(self.my_results_dir) > 0 and not self.use_shared_cnn:
          print("   Old directory...")
          my_results_path = self.part_results_path + self.my_results_dir + '/'
          my_weights_path = my_results_path + 'weights/'
          my_output_path = my_results_path + 'output/'
          my_actual_results_dir = self.my_results_dir
      elif len(self.new_results_dir) > 0:
          print("   New directory... (name given)")
          my_actual_results_dir = self.new_results_dir
          my_results_dir = self.dataset + '/' + self.new_results_dir + '/'
          my_results_path = self.part_results_path + my_results_dir
          
          if os.path.exists(my_results_path):
              raise Exception("directory already exists %s" % my_results_path)
          
          my_weights_path = my_results_path + 'weights/'
          my_output_path = my_results_path + 'output/'
          os.mkdir(my_results_path)
          os.mkdir(my_weights_path)
          os.mkdir(my_output_path)
          
      else:
          print("   New directory...")
          for fid in range(100):
            my_actual_results_dir = self.modelnamekey + '%d' % fid
            my_results_dir = self.dataset + "/" + my_actual_results_dir + '/'
            my_results_path = self.part_results_path + my_results_dir
            if not os.path.exists(my_results_path):
                my_weights_path = my_results_path + 'weights/'
                my_output_path = my_results_path + 'output/'
                os.mkdir(my_results_path)
                os.mkdir(my_weights_path)
                os.mkdir(my_output_path)
                break
            
      self.my_actual_results_dir = my_actual_results_dir  
      self.my_results_path = my_results_path
      self.my_evaluation_path = my_results_path
      self.my_weights_path = my_weights_path
      self.my_output_path = my_output_path
            
      if self.my_weights is not None:
          self.my_shared_weights = self.my_weights_path + self.my_weights
      
      if self.use_shared_cnn:
          self.my_shared_weights = self.part_results_path + self.my_results_dir + '/weights/' + self.my_weights
      
   def get_detections_path(self):
       if len(self.my_input_dir) == 0:
          return
       
       self.my_input_path = self.results_path + self.my_input_dir + '/output/'
      
      
   def update_paths(self):
       print('Updating paths...')
       self.get_data_path()
       self.get_results_paths()
       self.get_detections_path()
       
       print('   results_path:', self.my_results_path)
      
   def __init__(self, newDir = True):
       self.newDir = newDir
       self.setBasicValues()
       
   def setBasicValues(self):
       # PATHS
       
       # constant
       self.home_path = os.path.expanduser('~') + '/'
       self.base_path = ''
       self.weights_path = ''
       self.move_path = ''
       
       # constant partial paths
       self.part_results_path = ''
       self.part_data_path  = ''
       
       # variable partial paths
       self.results_path = ''
       self.data_path = ''
       
       # input directories
       self.my_results_dir = ''
       self.new_results_dir = ''
       self.my_input_dir = ''       
       
       
       # full paths       
       self.my_input_path = ''
       
       self.my_results_path = ''
       self.my_weights_path = ''
       self.my_evaluation_path = ''
       self.my_output_path = ''
       
       self.move = False
       self.use_shared_cnn = False
       self.my_shared_weights = None
       
       #basics
       self.dataset = 'HICO'
       self.inputs  = None
       self.max_classes = None
       self.backbone = None
       self.use_mean = True
       
       #generator
       self.train_cfg = self.gen_config()
       self.test_cfg = self.gen_config()
       self.val_cfg = self.gen_config()
       self.xdim= None
       self.ydim= None
       self.cdim= None
       self.minIoU = 0.5
       self.testdata = 'genTest'
       
       self.winShape = (64, 64)
       self.flip_image = False
       self.order_of_dims = [0,1,2]
       self.par_order_of_dims = [0,2,3,1]
       self.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
       
       
       #model
       self.task = None
       self.pretrained_weights = True
       self.my_weights = None
       self.only_use_weights = False
       self.use_l2_reg = True
       self.weight_decay = 0.0005
       self.weight_decay_shared = 0.0
       
       #model compile
       self.optimizer = 'sgd'
       self.wp = 1
       
       #model callbacks
       self.patience = 100
       self.modelnamekey = ''
       self.epoch_splits = [20,40,60,80]
       self.init_lr = None
       self.include_eval = True
       self.include_validation = False
       self.checkpoint_interval = 5
       
       # model training
       self.epoch_begin = None
       self.epoch_end = None
       
       #fast data preprocesses
       self.rpn_uniform_sampling = False
       
   def rcnn_config(self):
       self.xdim=227
       self.ydim=227
       self.cdim=3
              
       # Basic stuff
       self.init_lr = 0.0001
       self.epoch_begin = 0
       self.epoch_end = 60
       self.epoch_splits = [30]
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
       self.pool_size = 5
       self.init_lr = 0.00001
       self.epoch_begin = 0
       self.epoch_end = 60
       self.optimizer = 'adam'
       self.backbone = 'alex'
       self.do_fast_hoi = False
       
       self.winShape = (64, 64)
       
       self.train_cfg.batch_size = 1
       self.val_cfg.batch_size = 1
       self.test_cfg.batch_size = 1
       
       self.rpn_regr_std = 4.0
       self.rpn_regr_std = [0.125, 0.125, 0.25, 0.25]
       self.det_regr_std = [8.0, 8.0, 4.0, 4.0]
       
       #rpn filters
       self.nb_shared_layers = 17
       self.rpn_stride = 16
       self.nb_rpn_proposals = 256
        
       self.anchor_sizes = [64, 128, 256, 512]
       self.anchor_sizes = [0.5, 2, 8, 32]
       self.anchor_sizes = [4, 8, 16, 32]
       self.anchor_ratios = [[1, 1], [1, 0.5], [0.5, 1]]
       self.anchor_ratios = [0.5, 1, 2]
       
       self.nb_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
        
       self.rpn_min_overlap = 0.3
       self.rpn_max_overlap = 0.7
       
       self.rpn_nms_max_boxes=300
       self.rpn_nms_overlap_thresh=0.7
       self.rpn_nms_overlap_thresh_test=0.5
        
       # detection filters
       self.detection_max_overlap = 0.5
       self.detection_min_overlap = 0.0
       self.nb_detection_rois = 128
       self.det_fg_ratio = 0.25
       
       self.det_nms_max_boxes=300
       self.det_nms_overlap_thresh=0.9
       self.det_nms_overlap_thresh_test=0.5
        
       # hoi filters
       self.hoi_max_overlap = 0.5
       self.hoi_min_overlap = 0.1
       self.hoi_nms_overlap_thresh=0.5
       
       self.nb_hoi_rois = 32
       
       self.hoi_only_pos = False
       self.hoi_pos_share  = int(self.nb_hoi_rois / 8 * 4)
       self.hoi_neg1_share = int(self.nb_hoi_rois / 8 * 1)
       self.hoi_neg2_share = int(self.nb_hoi_rois / 8 * 3)
        
       # model
#       self.nb_object_classes = 81
#       self.nb_hoi_classes = 600
       
   def get_args(self):
       try:
          argv = sys.argv[1:]
          opts, args = getopt.getopt(argv,"ab:c:d:e:f:g:hi:j:l:m:n:o:pq:r:s:tuw:x:z")
       except getopt.GetoptError:
          print('.py argument error')
          sys.exit(2)
     
#    augment, backbone, cfg_method, dataset, epoch_split, final_epoch, generator_type, input_roi_dir, learning_rate, model, nb_batches, optimizer, results_dir, start_epoch, transfor data, uniform_sampling, weighing, ma(x)_classes
       print('Parsing args...')
       for opt, arg in opts:
          print('  ', opt, arg)
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
              splits = arg.split(',')
              splits = [int(x) for x in splits]
              self.epoch_splits = splits
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
              self.my_input_dir = arg
          if opt == '-j':
              self.weight_decay = float(arg)
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
          if opt == '-p':
              self.hoi_only_pos = True
          if opt == '-q':
              self.new_results_dir = arg
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
          if opt == '-z':
              self.do_fast_hoi = True
              
   def set_class_weights(self, labels, imagesMeta):
       if self.wp >= 0: 
           return
       print('  Using class-specific weights!')
       stats, counts = utils.getLabelStats(imagesMeta, labels)
       p = counts / sum(counts)
       wp = 1 / p
       self.wp = wp
