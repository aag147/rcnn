# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""

import keras
from keras import backend as K
import os
import losses
import numpy as np

from keras.models import load_model, Model
from keras.optimizers import SGD, Adam
from keras.utils.generic_utils import get_custom_objects

import detection.models.models as models,\
       layers
K.set_image_dim_ordering('tf')


class AllModels:
    def __init__(self, cfg, mode='train', do_rpn=False, do_det=False, do_hoi=False):
        self.mode = mode
        self.cfg = cfg
        
        self.do_rpn = do_rpn
        self.do_det = do_det
        self.do_hoi = do_hoi
        
        self.model_rpn = None
        self.model_det = None
        self.model_hoi = None
        
        self.nb_models = 0
        
        assert mode=='test' or np.sum([self.do_rpn, self.do_det, self.do_hoi])==1, 'Use only one model when training'
        
        if self.mode == 'train' and not cfg.use_shared_cnn and not cfg.only_use_weights and cfg.my_shared_weights is not None:
            self.load_models()
        else:
            self.create_models()
            self.load_weights()

    def save_model(self, saveShared=False):
        cfg = self.cfg
        
        if self.do_rpn:
            model = self.model_rpn
        if self.do_det:
            model = self.model_det
        if self.do_hoi:
            model = self.model_hoi
                    
        for i in range(10):
            modelpath = cfg.my_weights_path + 'model-theend%d.h5' % i
            weightspath = cfg.my_weights_path + 'weights-theend%d.h5' % i
            if not os.path.exists(modelpath):
                model.save(modelpath)
                model.save_weights(weightspath)
                
                if saveShared:
                    shared_cnn = Model(model.input, model.layers[17].output)
                    shared_cnn.save(cfg.my_weights_path + 'shared_model%d.h5' % i)
                    shared_cnn.save_weights(cfg.my_weights_path + 'shared_weights%d.h5' % i)
                
                break
        
    def compile_models(self):
        if self.mode == 'test':
            return
        
        cfg = self.cfg
        
        print('Compiling models...')
        if cfg.optimizer == 'adam':
            print('   Opt.:', 'Adam')
            opt = Adam(lr = cfg.init_lr)
        else:
            print('   Opt.:', 'SGD')
            opt = SGD(lr = cfg.init_lr, momentum = 0.9, decay = 0.0, nesterov=False)
        print('   Learning rate:', cfg.init_lr)
        if self.do_rpn:
            if cfg.rpn_uniform_sampling:
                print('   Uniform anchor sampling')
            else:
                print('   Non-Uniform anchor sampling')
            model = self.model_rpn
            my_losses = [losses.rpn_loss_cls(cfg.nb_anchors), losses.rpn_loss_regr(cfg.nb_anchors)]
            my_metrics = None
        if self.do_det:
            model = self.model_det
            my_losses=[losses.class_loss_cls, losses.class_loss_regr(cfg.nb_object_classes-1)]
            my_metrics={'det_out_class':'categorical_accuracy'}
        if self.do_hoi:
            model = self.model_hoi
            my_losses = [losses.hoi_loss_cls(cfg.wp)] 
            my_metrics = None
            
        model.compile(optimizer=opt, loss=my_losses, metrics=my_metrics)
        
        
    def get_models(self):
        return self.model_rpn, self.model_det, self.model_hoi
    
    def _load_shared_weights(self, model):
        cfg = self.cfg
        assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
        model.load_weights(cfg.my_shared_weights, by_name=True)
        # Only train unique layers
        for i, layer in enumerate(model.layers):
            layer.trainable = False
            if i == cfg.nb_shared_layers:
                break    
        return model
    
    def load_weights(self):
        cfg = self.cfg
        
        
        if type(cfg.my_weights)==str and len(cfg.my_weights)>0:
            print('Loading my weights...')
            
            if self.do_rpn:    
                rpn_before = self.model_rpn.layers[11].get_weights()[0][0,0,0,0]
                if self.mode == 'test' and self.nb_models > 1:
                    print('   Loading test RPN weights...')
                    path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir + '/weights/' + cfg.my_weights
                    assert os.path.exists(path), 'invalid path: %s' % path
                    self.model_rpn.load_weights(path, by_name=False)
                
                elif cfg.use_shared_cnn:
                    print('   Loading shared train RPN weights...')
                    self.model_rpn = self._load_shared_weights(self.model_rpn)
                    
                else:
                    print('   Loading train RPN weights...')
                    assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
                    self.model_rpn.load_weights(cfg.my_shared_weights) 
                    
                rpn_after = self.model_rpn.layers[11].get_weights()[0][0,0,0,0]
                assert rpn_before != rpn_after, 'RPN weights have not been loaded'
                
            
            if self.do_det:
                det_before = self.model_det.layers[4].get_weights()[0][0,0]
                if self.mode == 'test' and self.nb_models > 1:
                    print('   Loading test DET weights...')
                    path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir + '/weights/' + cfg.my_weights
                    assert os.path.exists(path), 'invalid path: %s' % path
                    self.model_det.load_weights(path, by_name=True)
                    
                elif cfg.use_shared_cnn:
                    print('   Loading shared train DET weights...')
                    self.model_det = self._load_shared_weights(self.model_det)
                    
                else:
                    print('   Loading train DET weights...')
                    assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
                    self.model_det.load_weights(cfg.my_shared_weights) 
                
                det_after = self.model_det.layers[4].get_weights()[0][0,0]
                assert det_before != det_after, 'DET weights have not been loaded'
            
            if self.do_hoi:
                hoi_before = self.model_hoi.layers[11].get_weights()[0][0,0,0,0]
                if self.mode == 'test' and self.nb_models > 1:
                    print('   Loading test HOI weights...')
                    path = cfg.part_results_path + 'HICO/hoi5c/weights/' + cfg.my_weights
                    assert os.path.exists(path), 'invalid path: %s' % path
                    self.model_hoi.load_weights(path, by_name=False)
                
                elif cfg.use_shared_cnn:
                    print('   Loading shared train HOI weights...')
                    self.model_hoi = self._load_shared_weights(self.model_hoi)
                    
                else:
                    print('   Loading train HOI weights...')
                    assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
                    self.model_hoi.load_weights(cfg.my_shared_weights) 
                
                hoi_after = self.model_hoi.layers[11].get_weights()[0][0,0,0,0]
                assert hoi_before != hoi_after, 'HOI weights have not been loaded'
        
    
            if self.do_rpn:
                rpn_final = self.model_rpn.layers[11].get_weights()[0][0,0,0,0]
                assert rpn_after == rpn_final, 'RPN weights have been overwritten'
            
            if self.do_det:
                det_final = self.model_det.layers[4].get_weights()[0][0,0]            
                assert det_after == det_final, 'DET weights have been overwritten'


    def load_models(self):
        cfg = self.cfg
        
        if self.do_rpn:
            print('Loading RPN model...')        
            loss_cls = losses.rpn_loss_cls(cfg.nb_anchors)
            loss_rgr = losses.rpn_loss_regr(cfg.nb_anchors)
            
            get_custom_objects().update({"rpn_loss_cls_fixed_num": loss_cls})
            get_custom_objects().update({"rpn_loss_regr_fixed_num": loss_rgr})
            
            assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
            self.model_rpn = load_model(cfg.my_shared_weights)


        if self.do_det:        
            print('Loading DET model...')        
            loss_cls = losses.class_loss_cls
            loss_rgr = losses.class_loss_regr(cfg.nb_object_classes-1)
            
            get_custom_objects().update({"class_loss_cls": loss_cls})
            get_custom_objects().update({"class_loss_regr_fixed_num": loss_rgr})
            
            assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
            self.model_det = load_model(cfg.my_shared_weights)
            
        if self.do_hoi:
            print('Loading HOI model...')        
            loss_cls = losses.hoi_loss_cls
            get_custom_objects().update({"hoi_loss_cls_fixed_num": loss_cls})
            
            assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
            self.model_hoi = load_model(cfg.my_shared_weights) 
            


    def create_models(self):
        cfg = self.cfg
        
        if self.mode=='test':
            print('Creating test models....')
        else:
            print('Creating train models....')
    
    
        ########################
        ###### Parameters ######
        ########################
        nb_anchors = cfg.nb_anchors
        pool_size = cfg.pool_size
        nb_object_classes = cfg.nb_object_classes
        nb_hoi_classes =cfg.nb_hoi_classes
        print('   Obj. classes:', nb_object_classes)
        print('   HOI classes:', nb_hoi_classes)
    
        ########################
        ##### Input shapes #####
        ########################    
        image_shape = (None, None, 3)
        roi_shape = (None, 5)
        human_shape = (None, 5)
        object_shape = (None, 5)
        interaction_shape = (None, cfg.winShape[0], cfg.winShape[1], 2)
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
        
        interaction_input = keras.layers.Input(
            shape=interaction_shape,
            name="input_interaction"
        )
        
        features_input = keras.layers.Input(
            shape=features_shape,
            name="input_features"
        )        
        
        img_hoi_input = keras.layers.Input(
            shape=image_shape,
            name='input_image'
        )
        
        ########################
        ####### Backbone #######
        ########################
        output_features = models.VGG16(cfg)(img_input)     
        output_features_hoi = models.VGG16(cfg)(img_hoi_input)
        
        ########################
        ######### RPN ##########
        ########################
        if self.do_rpn:
            self.nb_models += 1
            
            rpn_inputs = [
                img_input
            ]
            
            rpn_features = layers.rpn(cfg)([
                 output_features
            ])
            
            x_class = keras.layers.Conv2D(
                filters=nb_anchors,
                kernel_size=(1, 1),
                activation='sigmoid',
                kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                name='rpn_out_class'
            )(rpn_features)
            
            x_deltas = keras.layers.Conv2D(
                filters=nb_anchors * 4,
                kernel_size=(1, 1), 
                activation='linear', 
                kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                name='rpn_out_regress'
            )(rpn_features)
            
            if self.mode=='test':
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
            
            self.model_rpn = keras.models.Model(inputs=rpn_inputs, outputs=rpn_outputs)
            
            # Only train from conv3_1
            print('   Freezing first few layers...')
            for i, layer in enumerate(self.model_rpn.layers):
                layer.trainable = False
                if i > 6:
                    break
        
        ########################
        ###### Detection #######
        ########################
        if self.do_det:
            self.nb_models += 1
            
            if self.mode=='test':    
                detection_inputs = [
                    features_input,
                    roi_input
                ]
            else:
                detection_inputs = [
                    img_input,
                    roi_input
                ]
                
            pool_features_input = features_input if self.mode=='test' else output_features
                
            object_rois = layers.RoiPoolingConv(
                pool_size=pool_size,
            )([
                pool_features_input,
                roi_input
            ])
            
            object_features = layers.fullyConnected(
                cfg,
                stream = 'det'
            )([
                object_rois
            ])
            
            
            object_scores = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=nb_object_classes,
                    activation='softmax',
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                ),
                name="det_out_class"
            )(object_features)
            
            object_deltas = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=4 * (nb_object_classes - 1),
                    activation="linear",
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.001),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                ),
                name="det_out_regress"
            )(object_features)
    
            detection_outputs = [
                object_scores,
                object_deltas
            ]
            
            self.model_det = keras.models.Model(inputs=detection_inputs, outputs=detection_outputs)
    
            # Only train from conv3_1
            for i, layer in enumerate(self.model_det.layers):
                layer.trainable = False
                if i > 6:
                    break

        ########################
        ######### HOI ##########
        ########################    
        if self.do_hoi:
            self.nb_models += 1
            
            hoi_inputs = [
                img_hoi_input,
                human_input,
                object_input,
                interaction_input
            ]
            
            ## HUMAN ##
            hoi_human_rois = layers.RoiPoolingConv(
                pool_size=pool_size
            )([
                output_features_hoi,
                human_input
            ])
            
            hoi_human_features = layers.fullyConnected(
                cfg,
                stream = 'human'
            )([
                hoi_human_rois
            ])
    
            hoi_human_scores = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=1 * nb_hoi_classes,
                    activation=None,
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                ),
                name="scores4human"
            )(hoi_human_features)
                
            ## OBJECT ##
            hoi_object_rois = layers.RoiPoolingConv(
                pool_size=pool_size
            )([
                output_features_hoi,
                object_input
            ])
            
            hoi_object_features = layers.fullyConnected(
                cfg,
                stream = 'object'
            )([
                hoi_object_rois
            ])
    
            hoi_object_scores = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=1 * nb_hoi_classes,
                    activation=None,
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                ),
                name="scores4object"
            )(hoi_object_features)
                
                
            ## INTERACTION ##
            hoi_pattern_features = layers.pairwiseStream(
                cfg = cfg
            )([
                interaction_input
            ])
            hoi_pattern_scores = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=1 * nb_hoi_classes,
                    activation=None,
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                ),
                name = 'scores4pattern'
            )(hoi_pattern_features)
                
            ## FINAL ##
            hoi_score = keras.layers.Add()([hoi_human_scores, hoi_object_scores, hoi_pattern_scores])
            
            hoi_final_score = keras.layers.Activation(
                "sigmoid",
                name="hoi_out_class"
            )(hoi_score)
    
            
            if self.mode=='test':    
                hoi_outputs = [
                    hoi_final_score,
                    human_input,
                    object_input
                ]
            else:
                hoi_outputs = [
                    hoi_final_score
                ]
            
            self.model_hoi = keras.models.Model(inputs=hoi_inputs, outputs=hoi_outputs)


        ########################
        ######### ALL ##########
        ########################   
#        model_all = keras.models.Model([img_input,roi_input], rpn_outputs + detection_outputs)
