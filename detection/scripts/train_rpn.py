# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
import sys 
sys.path.append('../../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')

import numpy as np
import keras

import utils,\
       extract_data,\
       methods,\
       losses,\
       callbacks,\
       filters_helper as helper
from detection_generators import DataGenerator
    
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.optimizers import SGD, Adam

if True:
    # meta data
    data = extract_data.object_data()
    
    # config
    cfg = data.cfg
    cfg.fast_rcnn_config()



# models
model_rpn, model_detection, model_hoi = methods.get_hoi_rcnn_models(cfg)


opt = SGD(lr = 0.001, momentum = 0.9, decay = 0.0, nesterov=False)

model_rpn.compile(optimizer=opt,
                  loss=[losses.rpn_loss_cls(cfg.nb_anchors), losses.rpn_loss_regr(cfg.nb_anchors)], metrics=['accuracy'])

# data
#genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train')
genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val')
#genTest = DataGenerator(imagesMeta = data.testGTMeta, cfg=cfg, data_type='test')  


# train
callbacks = [callbacks.MyModelCheckpointInterval(cfg), \
             callbacks.MyLearningRateScheduler(cfg), \
             callbacks.SaveLog2File(cfg), \
             callbacks.PrintCallBack()]

model_rpn.fit_generator(generator = genVal.begin(), \
            steps_per_epoch = genVal.nb_batches, \
            epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=callbacks)
