# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""

import keras
from keras import backend as K


import detection.models.models as models,\
       layers
K.set_image_dim_ordering('tf')

def get_hoi_rcnn_models(cfg, mode='train'):
        ########################
        ###### Parameters ######
        ########################
        nb_anchors = cfg.nb_anchors
        pool_size = cfg.pool_size
        nb_object_classes = cfg.nb_object_classes
        nb_hoi_classes =cfg.nb_hoi_classes
        print('nb classes', nb_object_classes)
    
        ########################
        ##### Input shapes #####
        ########################    
        image_shape = (None, None, 3)
        roi_shape = (None, 5)
        human_shape = (None, 5)
        object_shape = (None, 5)
        features_shape = (None, None, 512)
    
        ########################
        ######## Inputs ########
        ########################
        img_input = keras.layers.Input(
            shape=image_shape,
            name='input_image'
        )
        roi_input = keras.layers.Input(
            shape=roi_shape,
            name='input_roi'
        )
        human_input = keras.layers.Input(
            shape=human_shape,
            name="input_human"
        )
        
        object_input = keras.layers.Input(
            shape=object_shape,
            name="input_object"
        )
        
        features_input = keras.layers.Input(
            shape=features_shape,
            name="input_features"
        )        
        
        ########################
        ####### Backbone #######
        ########################
        output_features = models.VGG16(cfg.weights_path)(img_input)
        
        ########################
        ######### RPN ##########
        ########################
        rpn_inputs = [
            img_input
        ]
        
        rpn_features = layers.rpn([
             output_features
        ])
        
        x_class = keras.layers.Conv2D(
            filters=nb_anchors,
            kernel_size=(1, 1),
            activation='sigmoid',
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            name='rpn_out_class'
        )(rpn_features)
        
        x_deltas = keras.layers.Conv2D(
            filters=nb_anchors * 4,
            kernel_size=(1, 1), 
            activation='linear', 
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            name='rpn_out_regress'
        )(rpn_features)
        
        if mode=='test':
            rpn_outputs = [
                x_class,
                x_deltas,
                output_features
            ]
        else:
            rpn_outputs = [
                x_class,
                x_deltas
            ]
        
        model_rpn = keras.models.Model(inputs=rpn_inputs, outputs=rpn_outputs)
        
        # Only train from conv3_1
        for i, layer in enumerate(model_rpn.layers):
            layer.trainable = False
            if i > 6:
                break
        
        ########################
        ###### Detection #######
        ########################
        if mode=='test':    
            detection_inputs = [
                features_input,
                roi_input
            ]
        else:
            detection_inputs = [
                img_input,
                roi_input
            ]
            
        pool_features_input = features_input if mode=='test' else output_features
            
        object_rois = layers.RoiPoolingConv(
            pool_size=pool_size,
        )([
            pool_features_input,
            roi_input
        ])
        
        object_features = layers.fullyConnected(
            stream = 'det'
        )([
            object_rois
        ])
        
        
        object_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=nb_object_classes,
                activation='softmax',
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01)
            ),
            name="det_out_class"
        )(object_features)
        
        object_deltas = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=4 * (nb_object_classes - 1),
                activation="linear",
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.001)
            ),
            name="det_out_regress"
        )(object_features)

        detection_outputs = [
            object_scores,
            object_deltas
        ]
        
        model_detection = keras.models.Model(inputs=detection_inputs, outputs=detection_outputs)

        # Only train from conv3_1
        for i, layer in enumerate(model_detection.layers):
            layer.trainable = False
            if i > 6:
                break

        ########################
        ######### HOI ##########
        ########################       
        hoi_inputs = [
            img_input,
            human_input,
            object_input
        ]
        
        hoi_human_rois = layers.RoiPoolingConv(
            pool_size=pool_size
        )([
            output_features,
            human_input
        ])
        
        hoi_human_features = layers.fullyConnected(
            stream = 'human'
        )([
            hoi_human_rois
        ])

        hoi_human_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * nb_hoi_classes,
                activation=None,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01)
            ),
            name="scores4human"
        )(hoi_human_features)
            
        hoi_object_rois = layers.RoiPoolingConv(
            pool_size=pool_size
        )([
            output_features,
            object_input
        ])
        
        hoi_object_features = layers.fullyConnected(
            stream = 'object'
        )([
            hoi_object_rois
        ])

        hoi_object_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * nb_hoi_classes,
                activation=None,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01)
            ),
            name="scores4object"
        )(hoi_object_features)
            
        hoi_score = keras.layers.Add()([hoi_human_scores, hoi_object_scores])
        
        
        hoi_final_score = keras.layers.Activation(
            "sigmoid",
            name="hoi_out_class"
        )(hoi_score)
        
        hoi_outputs = [
            hoi_final_score,
            human_input,
            object_input
        ]
        
        model_hoi = keras.models.Model(inputs=hoi_inputs, outputs=hoi_outputs)
        
        
        if mode=='test':
            return model_rpn, model_detection, model_hoi    

        ########################
        ######### ALL ##########
        ########################   
        model_all = keras.models.Model([img_input,roi_input], rpn_outputs + detection_outputs)
        
        return model_rpn, model_detection, model_hoi, model_all