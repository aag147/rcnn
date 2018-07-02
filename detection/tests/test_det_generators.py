# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
if True:
    import sys 
    sys.path.append('../../../')
    sys.path.append('../../')
    sys.path.append('../../shared/')
    sys.path.append('../models/')
    sys.path.append('../filters/')
    sys.path.append('../data/')
    
    import numpy as np
    
    import utils,\
           extract_data,\
           methods,\
           losses,\
           callbacks,\
           filters_helper as helper
    from det_generators import DataGenerator
    

    # meta data
    data = extract_data.object_data(False)
    
    # config
    cfg = data.cfg
    obj_mapping = data.class_mapping

    # data
    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)
#    genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=False)


genItr = genTrain.begin()
for batchidx in range(genTrain.nb_batches):
    out = next(genItr)
#    if batchidx+1 % 100 == 0:
    utils.update_progress_new(batchidx+1, genTrain.nb_batches, '')