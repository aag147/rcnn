# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:40:50 2018

@author: aag14
"""
if False:
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
#    genTrain = DataGenerator(imagesMeta = data.trainGTMeta, cfg=cfg, data_type='train', do_meta=False)
genVal = DataGenerator(imagesMeta = data.valGTMeta, cfg=cfg, data_type='val', do_meta=True)


genItr = genVal.begin()
for batchidx in range(genVal.nb_batches):
    [X,rois], y, imageMeta, imageDims, _ = next(genItr)
#    if batchidx+1 % 100 == 0:
    utils.update_progress_new(batchidx+1, genVal.nb_batches, '')
    
    import draw
    import filters_detection
    
    img = np.copy(X[0])
    img += cfg.PIXEL_MEANS
    img = img.astype(np.uint8)
    rois = filters_detection.unprepareInputs(rois, imageDims)
    
    draw.drawOverlapAnchors(img, rois[0], imageMeta, imageDims, cfg)
    
    break