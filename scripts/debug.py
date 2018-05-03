# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:30 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')

#import extractTUHOIData as tuhoi
#import extractHICOData as hico
import utils, image
from fast_generators import DataGenerator
from load_data import data

#from matplotlib import pyplot as plt
import cv2 as cv, numpy as np
import os
import sys
import time

#plt.close("all")

print('Loading data...')
# Load data
data = data(newDir=False)
cfg = data.cfg
cfg.fast_rcnn_config()

    
if True:
    # Create batch generators
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    genVal = DataGenerator(imagesMeta=data.valMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='val')
    genTest = DataGenerator(imagesMeta=data.testMeta, GTMeta = data.testGTMeta, cfg=cfg, data_type='test')  



if False:    
    i = 0
    start = time.time()
    print('Begin...')
    for sample in genTrain.begin():
#        utils.update_progress(i / genTrain.nb_batches)
        if i > genTrain.nb_batches:
            break
        i += 1
    end = time.time()
    print('End:', end - start)
  
    
def unnormCoords(box, shape):
    xmin = box[1] * shape[1]; xmax = box[3] * shape[1]
    ymin = box[0] * shape[0]; ymax = box[2] * shape[0]
    return [ymin, xmin, ymax, xmax]    

if False:
    # test tf crop_and_resize    
    from keras.models import Sequential, Model
    from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
        Input, merge
    from matplotlib import pyplot as plt

    import tensorflow as tf  
    from models import RoiPoolingConv
    import draw
    
    imagesMeta = data.trainMeta
    imagesID = list(imagesMeta.keys())
    imagesID.sort()
#    imageMeta = imagesMeta[imagesID[0]]
    path = cfg.data_path+'images/train/'
#    print(path + imageMeta['imageName'])
#    img = cv.imread(cfg.data_path+'images/train/' + imageMeta['imageName'])
#    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#    
#    rel = imageMeta['rels']['0']
#    prsBB = rel['prsBB']
#    objBB = rel['objBB']
    
    cfg.img_out_reduction = (16,16)
    cfg.pool_size = 7
    print('cfg', cfg.order_of_dims)
    
    idx = 0
    for sample in genTrain.begin():
        img = sample[0][0][idx]
        h   = sample[0][1][idx]
        o   = sample[0][2][idx]
        win = sample[0][3][idx]
        break
    print(img.shape)
    padding = np.zeros([300, 300, 3])
    padding[38:262,38:262,:] = img
    h[1] = (h[1] * 244 + 38) / 300
    h[2] = (h[2] * 244 + 38) / 300
    h[3] = h[3] * 244 / 300
    h[4] = h[4] * 244 / 300
    o[1] = (o[1] * 244 + 38) / 300
    o[2] = (o[2] * 244 + 38) / 300
    o[3] = o[3] * 244 / 300
    o[4] = o[4] * 244 / 300
    
#    shape = tuple([x//16 for x in img.shape[0:2]])
    img = cv.resize(padding, shape).astype(np.float32)
    boxes = np.array([h, o])
    imgs = np.expand_dims(img, axis=0)
    imgs = np.append(imgs, imgs, axis=0)
    
    
    draw.drawHOI(img, unnormCoords(h[1:], img.shape), unnormCoords(o[1:], img.shape))
#    draw.drawHOI(img[0], h, o)
    draw.drawImages(imagesID[0:1], imagesMeta, data.labels, path)
    
    cfg.pool_size = 7
    image_input = Input(shape=(None,None,3))
    boxes_input = Input(shape=(5,))
#    final_output = tf.image.crop_and_resize(image_input, boxes=boxes_input, box_ind=idxs_input, crop_size=(pool_size, pool_size))    
    roi_output = RoiPoolingConv(cfg)([image_input, boxes_input])
    model = Model(inputs=[image_input, boxes_input], outputs=roi_output)  
    
    pred = model.predict([imgs, boxes])
    
    f, spl = plt.subplots(2,2)
    spl = spl.ravel()
    spl[0].imshow(pred[0])
    spl[1].imshow(pred[1])
    
if False:
    # Save labels in file
    annotations = hico.getUniqueLabels(cfg)
    labels = []
    for annot in annotations:
        obj = annot.nname; pred = annot.vname; pred_ing = annot.vname_ing
        label = {'obj': obj, 'pred': pred, 'pred_ing': pred_ing}
        labels.append(label)
        
    utils.save_dict(labels, cfg.part_data_path + 'HICO_labels')
    labels = utils.load_dict(cfg.part_data_path + 'HICO_labels')
    
    stats, counts = utils.getLabelStats(trainMeta, labels)
    reduced_trainMeta, reduced_idxs = utils.reduceTrainData(trainMeta, counts, 25)
    reduced_testMeta = utils.reduceTestData(testMeta, reduced_idxs)
    reduced_labels = utils.idxs2labels(reduced_idxs, labels)
    reduced_stats, reduced_counts = utils.getLabelStats(reduced_trainMeta, reduced_labels)

if False:
    # Test if all images can be loaded and cropped successfully
    i = 0
    c = 0.0
    end = len(testMeta)
    for imageID, metaData in testMeta.items():
        oldPath = cfg.part_data_path + 'HICO_images/test/' + metaData['imageID']
        image = cv.imread(oldPath)
        if i / end > c:
#            print(c)
            c += 0.01
#        print(str(c) + ': ' + str(imageID), end='')
        sys.stdout.write('\r' + str(c) + ': ' + str(imageID))
        sys.stdout.flush()
        if image is None:
            print(imageID)
        for relID, rel in metaData['rels'].items():
#            print(imageID, relID)
            relCrops = utils.cropImageFromRel(rel['prsBB'], rel['objBB'], image)
            relCrops = utils.preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, (227,227))
        i += 1


if False:
    # Debug problems with images/bounding boxes (like a wrongly rotated image)
    imageID = 'HICO_train2015_00027301.jpg'
    imageMeta = trainMeta[imageID]
    oldPath = cfg.part_data_path + 'HICO_images/train/' + imageMeta['imageID']
    image = cv.imread(oldPath)
    i = 0
    for relID, rel in metaData['rels'].items():
#            print(imageID, relID)
        relCrops = utils.cropImageFromRel(rel['prsBB'], rel['objBB'], image)
        relCrops = utils.preprocessRel(relCrops['prsCrop'], relCrops['objCrop'], image, (227,227))
        i += 1

if False:
    # Plot images in range
    imagesID = list(trainMeta.keys())
    imagesID.sort()
    draw.drawImages(imagesID[26960:26969], trainMeta, labels, cfg.data_path+'_images/train/', False)

if True:
    # Test fast-generators by plotting data
    from fast_generators import DataGenerator
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    import matplotlib.pyplot as plt
    import draw
    i = 0
    idx = 0
    j = 0
    
    for sample in genTrain.begin():
#        utils.update_progress(j / len(data.trainMeta))
#        print(sample[1][idx])
        image = sample[0][0][idx]
        prsBB = sample[0][1][idx][0]
        objBB = sample[0][2][idx][0]
        y     = sample[1][idx]
        print(y)
        prsBB = unnormCoords(prsBB[1:], image.shape)
        objBB = unnormCoords(objBB[1:], image.shape)
        draw.drawHOI(image, prsBB, objBB)
        
        i += 1
        if i >= 5:
            break
        

if False:
    # Test fast-generators by plotting data
    from generators import DataGenerator
    genTrain = DataGenerator(imagesMeta=data.trainMeta, GTMeta = data.trainGTMeta, cfg=cfg, data_type='train')
    import matplotlib.pyplot as plt
    import draw
    i = 0
    idx = 0
    j = 0
    
    mu_counts = {}
    my_counts = {}
    for imageID, imageMeta in data.trainMeta.items():
        if imageID not in mu_counts:
            mu_counts[imageID] = 0
            mu_counts[imageID] += len(imageMeta['rels'])
    
    for sample in genTrain.begin():
#        utils.update_progress(j / len(data.trainMeta))
#        print(sample[1][idx])
        image = sample[0][0][idx]
        prsBB = sample[0][1][idx]
        objBB = sample[0][2][idx]
        
        for imageID in sample[1]:
            if imageID not in my_counts:
                my_counts[imageID] = 0
            my_counts[imageID] += 1
#        print(image.shape)
#        print(prsBB)
#        print(objBB)
#        draw.drawHOI(image, prsBB, objBB)
        i += 1
        if genTrain.nb_batches == i:
            break

if False:
    # Count instances of classes
    for imageID, mucounts in mu_counts.items():
        mu = mucounts
        if imageID not in my_counts:
            print(imageID, mu)
            continue
        my = my_counts[imageID]
        if mu != my:
            print(imageID, mu, my)


if False:
    # Test generators by plotting data
    genTrain = DataGenerator(imagesMeta=trainMeta, cfg=cfg, data_type='train')
    i = 0
    idx = 25
    j = 0
    for sample in genTrain.begin():
        utils.update_progress(j / len(trainMeta))
#        print(len(sample))
        for ys in sample[1]:
            j += 1
        continue
        print(np.argmax(sample[1][idx]))
        win = sample[0][2][idx]
        prs = sample[0][0][idx].transpose([1,2,0])
        obj = sample[0][1][idx].transpose([1,2,0])
        f, spl = plt.subplots(2,2)
        spl = spl.ravel()
        spl[0].imshow(win[0], cmap=plt.cm.gray)
        spl[1].imshow(win[1], cmap=plt.cm.gray)
        spl[2].imshow(prs)
        spl[3].imshow(obj)
        i += 1
        if i == 5:
            break




if False:
    from extractHICOData import combineSimilarBBs
    # Check tu-ppmi images manually
#    oldStats, oldCounts = utils.getLabelStats(trainMeta, labels)
#    newStats, newCounts = utils.getLabelStats(newTrainMeta, labels)
    
    
    imagesMeta = trainMeta
    imagesID = list(imagesMeta.keys())
    imagesID.sort()
    i = 50
    n = 4
    imagesID = imagesID[i*n+2:i*n+n+2]
#    tmpMeta = {imageID: imageMeta for imageID, imageMeta in imagesMeta.items() if imageID in imagesID}
#    somethingelse = combineSimilarBBs(tmpMeta, labels, 0.4)
    print("i",i*n)
    draw.drawImages(imagesID, imagesMeta, labels, cfg.data_path +'images/train/', False)
    
    
if False:
    # Change tu-ppmi dict
    new_imagesMeta = {}
    for imageID, imageMeta in data.testMeta.items():
        new_rels = {}
        for relID, rel in imageMeta['rels'].items():
            [label] = rel['labels']
            new_rel = {'objBB':rel['objBB'], 'prsBB':rel['prsBB'], 'label':label}
            new_rels[relID] = new_rel
        new_imagesMeta[imageID] = {'imageName': imageMeta['imageName'], 'rels': new_rels}
        
    imagesID = list(new_imagesMeta.keys())
    imagesID.sort()
    utils.save_dict(new_imagesMeta, cfg.data_path + 'testreal')
#    new_imagesMeta = utils.load_dict(cfg.data_path + 'test')
#    draw.drawImages(imagesID[50:59], new_imagesMeta, labels, cfg.data_path+'images/test/', False)
#
#imageID = trainMeta[1]
#X, y = next(genTrain)
#imageMeta = imagesMeta[imageID]
#rel = imageMeta['rels'][0]
#image = images[imageID]
#f, spl = plt.subplots(2,2)
#spl = spl.ravel()
#spl[0].imshow(X[2][0], cmap=plt.cm.gray)
#spl[1].imshow(X[2][1], cmap=plt.cm.gray)
#spl[2].imshow(X[0])
#spl[3].imshow(X[1])
#
#print(np.unique(res['pairwise'][0]))
#print(np.unique(res['pairwise'][1]))

#mean = [103.939, 116.779, 123.68]
#shape = (224, 224)
#images = pp.loadImages(imagesID, imagesMeta, url+"images/")
#
#imagesCrops = pp.cropImagesFromRels(imagesID, imagesMeta, images)
#resizedImagesCrops = pp.preprocessCrops(imagesCrops, shape, mean)
#[trainID, testID] = pp.splitData(imagesID)
#
#pdata.drawImages([imageID], imagesMeta, url+'images/', False)
#
#pdata.drawCrops(imagesID, imagesMeta, imagesCrops, images)
