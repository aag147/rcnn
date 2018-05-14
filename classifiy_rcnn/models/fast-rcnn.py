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
    Parameters
    ----------
    input_shape : A shape tuple (integer) without the batch dimension.
        For example:
            `input_shape=(224, 224, 3)`
        specifies that the input are batches of $224 × 224$ RGB images.
        Likewise:
            `input_shape=(224, 224)`
        specifies that the input are batches of $224 × 224$ grayscale
        images.
    categories : An array-like with shape:
            $$(categories,)$$.
        For example:
            `categories=["circle", "square", "triangle"]`
        specifies that the detected objects belong to either the
        “circle,” “square,” or “triangle” category.
    anchor_aspect_ratios : An array-like with shape:
            $$(aspect_ratios,)$$
        used to generate anchors.
        For example:
            `aspect_ratios=[0.5, 1., 2.]`
        corresponds to 1:2, 1:1, and 2:1 respectively.
    anchor_base_size : Integer that specifies an anchor’s base area:
            $$base_area = base_size^{2}$$.
    anchor_scales : An array-like with shape:
            $$(scales,)$$
        used to generate anchors. A scale corresponds to:
            $$area_{scale}=\sqrt{\frac{area_{anchor}}{area_{base}}}$$.
    anchor_stride : A positive integer
    backbone :
    dense_units : A positive integer that specifies the dimensionality of
        the fully-connected layers.
        The fully-connected layers are the layers that precede the
        fully-connected layers for the classification, regression and
        segmentation target functions.
        Increasing the number of dense units will increase the
        expressiveness of the network and consequently the ability to
        correctly learn the target functions, but it’ll substantially
        increase the number of learnable parameters and memory needed by
        the model.
    mask_shape : A shape tuple (integer).
    maximum_proposals : A positive integer that specifies the maximum
        number of object proposals returned from the model.
        The model always return an array-like with shape:
            $$(maximum_proposals, 4)$$
        regardless of the number of object proposals returned after
        non-maximum suppression is performed. If the number of object
        proposals returned from non-maximum suppression is less than the
        number of objects specified by the `maximum_proposals` parameter,
        the model will return bounding boxes with the value:
            `[0., 0., 0., 0.]`
        and scores with the value `[0.]`.
    minimum_size : A positive integer that specifies the maximum width
        or height for each object proposal.
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
#            input_human_bounding_boxes,
            input_object_bounding_boxes
        ]
        
        ########################
        ####### Backbone #######
        ########################
        
        output_features = models.VGG16_keras()(input_image)


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
                kernel_initializer="uniform",
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
                kernel_initializer="uniform",
                name="scores2object"
            )
        )(object_output)


        ########################
        ######## Output ########
        ########################        
        score = keras.layers.Add()([human_scores, object_scores])
        
        
        output = keras.layers.Activation("sigmoid",name="predictions")(object_scores)

        outputs = [
#            human_rois,
            object_rois,
            output_features
        ]

        super(RCNN, self).__init__(inputs, output)

    

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
    
if False:
    # Train model
    print('Training model...')
    log = callbacks.LogHistory()
    my_callbacks = [log, \
             callbacks.MyModelCheckpointInterval(cfg), \
             callbacks.MyLearningRateScheduler(cfg), \
             callbacks.SaveLog2File(cfg), \
             callbacks.PrintCallBack()
#             callbacks.EvaluateTest(genTest, metrics.EvalResults, cfg)
    ]
       

    model.fit_generator(generator = genTrain.begin(), \
        steps_per_epoch = genTrain.nb_batches, \
        epochs = cfg.epoch_end, initial_epoch=cfg.epoch_begin, callbacks=my_callbacks)
    
    print('Path:', cfg.my_results_path)

    
    
if False:
    iterGen = genTest.begin()
    import matplotlib.pyplot as plt
    for i in range(genTest.nb_batches):
        batch, y = next(iterGen)
        image = batch[0][0]
        hbb = batch[1][0,0,1:]
#        obb = batch[2][0,0,1:]
        y_hat = model.predict_on_batch(x=batch)
        if 0 in y[:,:,0]:
            print('y', y)
            print('y_', y_hat)

#        f, spl = plt.subplots(2,2)
#        spl = spl.ravel()
#        spl[0].imshow(h[0,0,:,:,0])
#        spl[1].imshow(o[0,0,:,:,0])
#        spl[2].imshow(i[0,:,:,0])
#        spl[3].imshow(image)
        
#        shape = image.shape
#        hbbx = [hbb[1],hbb[3],hbb[3],hbb[1],hbb[1]]
#        hbbx = [x*shape[1] for x in hbbx]
#        hbby = [hbb[0],hbb[0],hbb[2],hbb[2],hbb[0]]
#        hbby = [y*shape[0] for y in hbby]
#        
#        obbx = [obb[1],obb[3],obb[3],obb[1],obb[1]]
#        obbx = [x*shape[1] for x in obbx]
#        obby = [obb[0],obb[0],obb[2],obb[2],obb[0]]
#        obby = [y*shape[0] for y in obby]
#        spl[3].plot(hbbx, hbby, c='blue')
#        spl[3].plot(obbx, obby, c='red')
          