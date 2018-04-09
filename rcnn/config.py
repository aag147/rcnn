# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:08:31 2018

@author: aag14
"""

class config:
   def __init__(self, url2Images, nb_classes):
       #basics
       self.url2Images = url2Images
       self.nb_classes = nb_classes
       
       #generator
       self.xdim=227
       self.ydim=227
       self.cdim=3
       
       #model
       self.task = 'multi-class'
       
       #model callbacks
       self.patience = 0
       self.modelnamekey = ''
       self.epoch_split = 5
       self.init_lr = 0.001
       
       # model training
       self.epoch_begin = 0
       self.epoch_end = 5

    