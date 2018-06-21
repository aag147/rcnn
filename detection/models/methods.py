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
            self.compile_models()

    def save_model(self, saveShared=False):
        print('Saving model...')
        
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
                    print('   Saving shared CNN model...')
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
    
    
    def _load_test_weights(self, model, path, by_name = False):
        print('   -loading test weights...')
        cfg = self.cfg
        # Check path to weights
        if path[-1] == 'f':
            if not os.path.exists(path):
                path = path[:-1]
            elif model.name != 'rpn':
                by_name = True
        path += '/weights/' + cfg.my_weights
        print('   Weights path:', path)
        assert os.path.exists(path), 'invalid path: %s' % path
        
        # Load weights
        weights_before = model.layers[4].get_weights()[0][0,0] if by_name else model.layers[11].get_weights()[0][0,0,0,0]
        model.load_weights(path, by_name=False)        
        weights_after  = model.layers[4].get_weights()[0][0,0] if by_name else model.layers[11].get_weights()[0][0,0,0,0]
        assert weights_before != weights_after, 'weights have not been loaded'
        
        return model, weights_before, weights_after
    
    def _load_shared_weights(self, model):
        print('   -loading shared weights...')
        cfg = self.cfg
        # Check path to weights
        path = cfg.my_shared_weights
        print('   Weights path:', path)
        assert os.path.exists(path), 'invalid path: %s' % path
        
        # Load weights
        weights_before = model.layers[11].get_weights()[0][0,0,0,0]
        model.load_weights(path, by_name=True)
        weights_after = model.layers[11].get_weights()[0][0,0,0,0]
        assert weights_before != weights_after, 'weights have not been loaded'
        
        # Only train unique layers
        for i, layer in enumerate(model.layers):
            layer.trainable = False
            if i == cfg.nb_shared_layers:
                break
        return model, weights_before, weights_after
    
    def _load_train_weights(self, model):
        print('   -loading train weights...')
        cfg = self.cfg
        # Check path to weights
        path = cfg.my_shared_weights
        print('   Weights path:', path)
        assert os.path.exists(path), 'invalid path: %s' % path
        
        # Load weights
        weights_before = model.layers[11].get_weights()[0][0,0,0,0]
        model.load_weights(path, by_name=False)
        weights_after = model.layers[11].get_weights()[0][0,0,0,0]
        assert weights_before != weights_after, 'weights have not been loaded'
        
        return model, weights_before, weights_after
        
    def load_weights(self):
        cfg = self.cfg
        
        
        if type(cfg.my_weights)==str and len(cfg.my_weights)>0:
            print('Loading my weights...')
            
            if self.do_rpn:    
                print('   Loading RPN weights...')
                if self.mode == 'test' and self.nb_models > 0:
                    path = cfg.part_results_path + "COCO/rpn" + cfg.my_results_dir
                    self.model_rpn, rpn_before, rpn_after = self._load_test_weights(self.model_rpn, path)
                elif cfg.use_shared_cnn:
                    self.model_rpn, rpn_before, rpn_after = self._load_shared_weights(self.model_rpn)
                else:
                    self.model_rpn, rpn_before, rpn_after = self._load_train_weights(self.model_rpn)
                
            
            if self.do_det:
                print('   Loading DET weights...')
                if self.mode == 'test' and self.nb_models > 0:
                    path = cfg.part_results_path + "COCO/det" + cfg.my_results_dir
                    self.model_det, det_before, det_after = self._load_test_weights(self.model_det, path)
                elif cfg.use_shared_cnn:
                    self.model_det, det_before, det_after = self._load_shared_weights(self.model_det)
                else:
                    self.model_det, det_before, det_after = self._load_train_weights(self.model_det)
            
            if self.do_hoi:
                print('   Loading HOI weights...')
                if self.mode == 'test' and self.nb_models > 0:
                    path = cfg.part_results_path + "HICO/hoi" + cfg.my_results_dir
                    self.model_hoi, hoi_before, hoi_after = self._load_test_weights(self.model_hoi, path)
                elif cfg.use_shared_cnn:
                    self.model_hoi, hoi_before, hoi_after = self._load_shared_weights(self.model_hoi)
                else:
                    self.model_hoi, hoi_before, hoi_after = self._load_train_weights(self.model_hoi)
                    
            if self.do_rpn:
                rpn_final = self.model_rpn.layers[11].get_weights()[0][0,0,0,0]
                assert rpn_after == rpn_final, 'RPN weights have been overwritten'
            
            if self.do_det:
                det_final = self.model_det.layers[4].get_weights()[0][0,0]
                if type(det_final) != float:
                    det_final = self.model_det.layers[11].get_weights()[0][0,0,0,0]
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
            self.model_rpn.name = 'rpn'


        if self.do_det:        
            print('Loading DET model...')        
            loss_cls = losses.class_loss_cls
            loss_rgr = losses.class_loss_regr(cfg.nb_object_classes-1)
            
            get_custom_objects().update({"class_loss_cls": loss_cls})
            get_custom_objects().update({"class_loss_regr_fixed_num": loss_rgr})
            
            get_custom_objects().update({"RoiPoolingConv": layers.RoiPoolingConv})
            
            assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
            self.model_det = load_model(cfg.my_shared_weights)
            self.model_det.name = 'det'
            
        if self.do_hoi:
            print('Loading HOI model...')        
            loss_cls = losses.hoi_loss_cls
            get_custom_objects().update({"hoi_loss_cls_fixed_num": loss_cls})
            
            assert os.path.exists(cfg.my_shared_weights), 'invalid path: %s' % cfg.my_shared_weights
            self.model_hoi = load_model(cfg.my_shared_weights) 
            self.model_hoi.name = 'hoi'
            


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
        
        img_det_input = keras.layers.Input(
            shape=image_shape,
            name='input_image'
        )
        
        img_hoi_input = keras.layers.Input(
            shape=image_shape,
            name='input_image'
        )
        
        ########################
        ####### Backbone #######
        ########################
        output_features = models.VGG16(cfg)(img_input) 
        
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
                bias_regularizer   = keras.regularizers.l2(cfg.weight_decay),
                name='rpn_out_class'
            )(rpn_features)
            
            x_deltas = keras.layers.Conv2D(
                filters=nb_anchors * 4,
                kernel_size=(1, 1), 
                activation='linear', 
                kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                bias_regularizer   = keras.regularizers.l2(cfg.weight_decay),
                name='rpn_out_regress'
            )(rpn_features)
            
            if self.mode=='test' and cfg.use_shared_cnn:
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
            self.model_rpn.name = 'rpn'
            
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
            output_features_det = models.VGG16(cfg)(img_det_input)
            
            self.nb_models += 1
            
            if self.mode=='test' and cfg.use_shared_cnn:    
                detection_inputs = [
                    features_input,
                    roi_input
                ]
            else:
                detection_inputs = [
                    img_det_input,
                    roi_input
                ]
                
            pool_features_input = features_input if self.mode=='test' and cfg.use_shared_cnn else output_features_det
                
            object_rois = layers.RoiPoolingConv(
                pool_size=pool_size,
            )([
                pool_features_input,
                roi_input
            ])
            
            object_features = layers.fullyConnected(
                cfg,
                stream = 'det',
                use_dropout = cfg.weight_decay == 0.0
            )([
                object_rois
            ])
            
            
            object_scores = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=nb_object_classes,
                    activation='softmax',
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                    bias_regularizer   = keras.regularizers.l2(cfg.weight_decay)
                ),
                name="det_out_class"
            )(object_features)
            
            object_deltas = keras.layers.TimeDistributed(
                keras.layers.Dense(
                    units=4 * (nb_object_classes - 1),
                    activation="linear",
                    kernel_initializer = keras.initializers.RandomNormal(stddev=0.001),
                    kernel_regularizer = keras.regularizers.l2(cfg.weight_decay),
                    bias_regularizer   = keras.regularizers.l2(cfg.weight_decay)
                ),
                name="det_out_regress"
            )(object_features)
    
            detection_outputs = [
                object_scores,
                object_deltas
            ]
            
            self.model_det = keras.models.Model(inputs=detection_inputs, outputs=detection_outputs)
            self.model_det.name = 'det'
    
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
            
            output_features_hoi = models.VGG16(cfg)(img_hoi_input)
            
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
            self.model_hoi.name = 'hoi'

        ########################
        ######### ALL ##########
        ########################   
#        model_all = keras.models.Model([img_input,roi_input], rpn_outputs + detection_outputs)
