# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:39:35 2018

@author: aag14
https://github.com/broadinstitute/keras-rcnn/blob/master/keras_rcnn/models/_rcnn.py
"""



import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')
sys.path.append('../layers/')


import keras
import numpy

import models,\
       classifier,\
       losses,\
       callbacks,\
       utils,\
       metrics


class RCNN(keras.models.Model):
    """
    A Region-based Convolutional Neural Network (RCNN)
    """
    def __init__(
            self,
            input_shape,
            labels,
            cfg,
            backbone=None
    ):



        self.nb_classes = len(labels)

        ########################
        ######## Inputs ########
        ########################
        
        input_human_bounding_boxes = keras.layers.Input(
            shape=(None, 5),
            name="target_human_bounding_boxes"
        )
        
        input_object_bounding_boxes = keras.layers.Input(
            shape=(None, 5),
            name="target_object_bounding_boxes"
        )

        input_image = keras.layers.Input(
            shape=input_shape,
            name="target_image"
        )


        inputs = [
            input_image,
            input_human_bounding_boxes,
            input_object_bounding_boxes
        ]
        
        ########################
        ####### Backbone #######
        ########################
        
        
        ########################
        ####### HUMAN SM #######
        ########################
        weights = cfg.weights_path + "alexnet_weights.h5"
        modelPrs = models.AlexNet((3, 227, 227), weights, cfg.nb_classes, include='fc')
        modelObj = models.AlexNet((3, 227, 227), weights, cfg.nb_classes, include='fc')


        ########################
        ####### HUMAN SM #######
        ########################
        human_rois = classifier.RoiPoolingConv(
            pool_size=cfg.pool_size
        )([
            output_features,
            input_human_bounding_boxes
        ])
        
        human_output = classifier.fullyConnected([
            human_rois
        ])
        

        human_deltas = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=4 * self.nb_classes,
                activation="linear",
                kernel_initializer="zero",
                name="deltas2human"
            )
        )(human_output)

        human_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * self.nb_classes,
                activation=None,
                kernel_initializer="zero",
                name="scores2human"
            )
        )(human_output)
            
            
        ########################
        ####### OBJECT S #######
        ########################
        object_rois = classifier.RoiPoolingConv(
            pool_size=cfg.pool_size
        )([
            output_features,
            input_object_bounding_boxes
        ])
        
        
        object_output = classifier.fullyConnected([
            object_rois
        ])
        

        object_deltas = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=4 * self.nb_classes,
                activation="linear",
                kernel_initializer="zero",
                name="deltas2object"
            )
        )(object_output)

        object_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * self.nb_classes,
                activation=None,
                kernel_initializer="zero",
                name="scores2object"
            )
        )(object_output)


        ########################
        ######## Output ########
        ########################        
        score = keras.layers.Add()([human_scores, object_scores])
        
        
        output = keras.layers.Activation("sigmoid",name="predictions")(score)

        outputs = [
            human_rois,
            object_rois,
            output_features
        ]

        super(RCNN, self).__init__(inputs, score)

    

    def predict(self, gen, verbose=0, steps=None):
        x = [target_bounding_boxes,target_categories,x]
        return super(RCNN, self).predict(x, batch_size=1, verbose=0, steps=None)



from load_data import data
from fast_generators import DataGenerator
import numpy as np


np.seterr(all='raise')

#plt.close("all")

if True:
    # Load data
    print('Loading data...')
    data = data()
    cfg = data.cfg
    cfg.fast_rcnn_config()
    
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  

if False:    
    # Save config
    utils.saveConfig(cfg)
    utils.saveSplit(cfg, list(data.trainMeta.keys()), list(data.valMeta.keys()))
    
    # Create model
    print('Creating model...')
    model = RCNN(input_shape = (None, None, 3),
            labels=data.labels,
            cfg=cfg)
    opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, decay = 0.0, nesterov=False)
    loss = losses.weigthed_binary_crossentropy(cfg.wp,1)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
#if False:
    # Train model
    print('Training model...')
    log = callbacks.LogHistory()
    my_callbacks = [log, \
             callbacks.MyModelCheckpointInterval(cfg), \
             callbacks.MyLearningRateScheduler(cfg), \
             callbacks.SaveLog2File(cfg), \
             callbacks.PrintCallBack(), \
             callbacks.EvaluateTest(genTest, metrics.EvalResults, cfg)
    ]
       

    model.fit_generator(generator = genTrain.begin(), \
        steps_per_epoch = genTrain.nb_batches, \
        epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=my_callbacks)
    
    print('Path:', cfg.my_results_path)

