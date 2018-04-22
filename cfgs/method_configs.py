# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""
def rcnn_hoi_classes(cfg):
   # OVERWRITE   
   cfg.inputs = [1,1,1]
   cfg.task = 'multi-class'
   cfg.patience = 1000
   cfg.epoch_begin = 0
   cfg.epoch_end = 100
   cfg.epoch_splits = [50]
   cfg.init_lr = 0.0001
   cfg.wp = 20
   
   return cfg

def rcnn_hoi_labels(cfg):
   # OVERWRITE
   print("TYPE: RCNN HOI LABELS")   
   cfg.inputs = [1,1,1]
   cfg.task = 'multi-label'
   cfg.patience = 1000
   cfg.epoch_begin = 0
   cfg.epoch_end = 60
   cfg.epoch_splits = [30]
   cfg.init_lr = 0.0001
   cfg.wp = 10
   
   return cfg

def rcnn_h_labels(cfg):
   # OVERWRITE   
   cfg.inputs = [1,0,0]
   cfg.task = 'multi-label'
   cfg.patience = 1000
   cfg.epoch_begin = 0
   cfg.epoch_end = 60
   cfg.epoch_splits = [30]
   cfg.init_lr = 0.0001
   cfg.wp = 10
   
   return cfg

def rcnn_o_labels(cfg):
   # OVERWRITE   
   cfg.inputs = [0,1,0]
   cfg.task = 'multi-label'
   cfg.patience = 1000
   cfg.epoch_begin = 0
   cfg.epoch_end = 60
   cfg.epoch_splits = [30]
   cfg.init_lr = 0.0001
   cfg.wp = 10
   
   return cfg

def rcnn_i_labels(cfg):
   # OVERWRITE   
   cfg.inputs = [0,0,1]
   cfg.task = 'multi-label'
   cfg.patience = 1000
   cfg.epoch_begin = 0
   cfg.epoch_end = 60
   cfg.epoch_splits = [30]
   cfg.init_lr = 0.0001
   cfg.wp = 10
   
   return cfg
