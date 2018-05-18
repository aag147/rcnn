# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""

import keras

import models,\
       layers

def get_hoi_rcnn_models(cfg):
        ########################
        ###### Parameters ######
        ########################
        nb_anchors = cfg.nb_anchors
        pool_size = cfg.pool_size
        nb_object_classes = cfg.nb_object_classes
        nb_hoi_classes =cfg.nb_hoi_classes
    
        ########################
        ##### Input shapes #####
        ########################    
        image_shape = (None, None, 3)
        roi_shape = (None, 5)
        human_shape = (None, 5)
        object_shape = (None, 5)
    
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
        
        ########################
        ####### Backbone #######
        ########################
        output_features = models.VGG16_keras()(img_input)
        
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
            kernel_initializer='uniform',
            name='rpn_out_class'
        )(rpn_features)
        
        x_deltas = keras.layers.Conv2D(
            filters=nb_anchors * 4,
            kernel_size=(1, 1), 
            activation='linear', 
            kernel_initializer='zero', 
            name='rpn_out_regress'
        )(rpn_features)
        
        rpn_outputs = [
            x_class,
            x_deltas
        ]
        
        model_rpn = keras.models.Model(inputs=rpn_inputs, outputs=rpn_outputs)
        
        ########################
        ###### Detection #######
        ########################
        detection_inputs = [
            img_input,
            roi_input
        ]
            
        object_rois = layers.RoiPoolingConv(
            pool_size=pool_size
        )([
            output_features,
            roi_input
        ])
        
        object_features = layers.fullyConnected([
            object_rois
        ])
        
        object_deltas = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=4 * (nb_object_classes - 1),
                activation="linear",
                kernel_initializer="zero",
                name="det_out_class"
            )
        )(object_features)

        object_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * nb_object_classes,
                activation='softmax',
                kernel_initializer="uniform",
                name="det_out_regress"
            )
        )(object_features)
        
        detection_outputs = [
            object_scores,
            object_deltas
        ]
        
        model_detection = keras.models.Model(inputs=detection_inputs, outputs=detection_outputs)
        

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
        
        hoi_human_features = layers.fullyConnected([
            hoi_human_rois
        ])

        hoi_human_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * nb_hoi_classes,
                activation=None,
                kernel_initializer="uniform",
                name="scores4human"
            )
        )(hoi_human_features)
            
        hoi_object_rois = layers.RoiPoolingConv(
            pool_size=pool_size
        )([
            output_features,
            object_input
        ])
        
        hoi_object_features = layers.fullyConnected([
            hoi_object_rois
        ])

        hoi_object_scores = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units=1 * nb_hoi_classes,
                activation=None,
                kernel_initializer="uniform",
                name="scores4object"
            )
        )(hoi_object_features)
            
        hoi_score = keras.layers.Add()([hoi_human_scores, hoi_object_scores])
        
        
        hoi_final_score = keras.layers.Activation("sigmoid",name="predictions")(hoi_score)
        
        hoi_outputs = [
            hoi_final_score,
            human_input,
            object_input
        ]
        
        model_hoi = keras.models.Model(inputs=hoi_inputs, outputs=hoi_outputs)
        
        
        return model_rpn, model_detection, model_hoi