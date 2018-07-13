# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:04:10 2018

@author: aag14
"""

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


import sys 
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../../shared/')
sys.path.append('../models/')
sys.path.append('../filters/')
sys.path.append('../data/')


import extract_data

import methods

if True:
    # Load data
    data = extract_data.object_data()
    cfg = data.cfg

    Models = methods.AllModels(cfg, mode='test', do_rpn=True, do_det=True, do_hoi=True)
    model_rpn, model_det, model_hoi = Models.get_models()

memrpn = get_model_memory_usage(1, model_rpn)
memdet = get_model_memory_usage(1, model_det)
memhoi = get_model_memory_usage(1, model_hoi)