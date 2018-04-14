# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:06:44 2018

@author: aag14
"""


import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import os
import glob

def plotLosses(log):
    # summarize history for loss
    f, spl = plt.subplots(1)
    spl = spl.ravel()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def drawCrops(imagesID, imagesMeta, imagesCrops, images):
    f, spl = plt.subplots(2,2)
    spl = spl.ravel()
    i = 0
    for imageID in imagesID[0:2:4]:
        imageMeta = imagesMeta[imageID]
        image = images[imageID]
        rel = imageMeta['rels'][0]
        prsBB = rel['prsBB']
        objBB = rel['objBB']
        print(rel)
        relCrops = imagesCrops[imageID][0]
        prsCropFull = np.zeros_like(image)
        print(relCrops['prsCrop'].shape)
        prsCrop = cp.copy(relCrops['prsCrop'])
        prsCrop[0:2,:,:] = [255, 0, 0]; prsCrop[-2:,:,:] = [255, 0, 0]
        prsCrop[:,0:2,:] = [255, 0, 0]; prsCrop[:,-2:,:] = [255, 0, 0]
        prsCropFull[prsBB['ymin']:prsBB['ymax'], prsBB['xmin']:prsBB['xmax'], :] = prsCrop
    
        objCrop = cp.copy(relCrops['objCrop'])
        objCrop[0:2,:,:] = [0, 0, 255]; objCrop[-2:,:,:] = [0, 0, 255]
        objCrop[:,0:2,:] = [0, 0, 255]; objCrop[:,-2:,:] = [0, 0, 255]
        prsCropFull[objBB['ymin']:objBB['ymax'], objBB['xmin']:objBB['xmax'], :] = objCrop
        
        spl[i].imshow(image)
        spl[i+1].imshow(prsCropFull)
        #spl[i+1].imshow(objCrop)

def drawBoundingBox(bb):
    xmin = bb['xmin']; xmax = bb['xmax']
    ymin = bb['ymin']; ymax = bb['ymax']
    
    box = np.array([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]])
    return box

def drawImages(imagesID, imagesMeta, labels, url, imagesBadOnes):
    f, spl = plt.subplots(3,3)
    spl = spl.ravel()
    i = 0
    for imageID in imagesID:
        print(imageID)
        imageMeta = imagesMeta[imageID]
        image = cv.imread(url + imageMeta['imageName'])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # + '.JPEG'
        spl[i].imshow(image)
        if imagesBadOnes:
            imageBadOnes = imagesBadOnes[imageID]
            colours = ['#000000', '#ffffff']
            j = 0
            for relsBad in imageBadOnes:
                for rel in relsBad:
                    objBB = drawBoundingBox(rel['bb'])
                    spl[i].plot(objBB[0,:], objBB[1,:], c=colours[j])
                j += 1
                
        colours = [['#FE5757', '#e44e4e'], ['#57FE57', '#4ee44e'], ['#5757FE', '#4e4ee4'], ['#AB57FE', '#994ee4']]
        j = 0
        titlesub = {}
        for relID, rel in imageMeta['rels'].items():
            c = colours[j % 4]
            personBB = drawBoundingBox(rel['prsBB'])
            objectBB = drawBoundingBox(rel['objBB'])
            spl[i].plot(personBB[0,:], personBB[1,:], c=c[0])
            spl[i].plot(objectBB[0,:], objectBB[1,:], c=c[1])
            
            for idx in rel['labels']:
                name = labels[idx]
                pred = name['pred']
                obj = name['obj']
                if relID not in titlesub.keys():
                    titlesub[relID] = {}
                if obj not in titlesub[relID].keys():
                    titlesub[relID][obj] = []
                titlesub[relID][obj].append("["+str(relID)+"] " + pred)
            j += 1
            
        #spl[i].set_yticklabels([])
        #spl[i].set_xticklabels([]) 
        spl[i].axis('off')
        spl[i].text(0.5, -0.1, ("\n".join([", ".join(j)+' '+k for r, i in titlesub.items() for k, j in i.items()])), \
           horizontalalignment='center', verticalalignment='center', \
           transform=spl[i].transAxes)
        #spl[i].set_title(", ".join(titlesub) + ": %s" % (imageID))
        i += 1
    f.tight_layout(rect=[0, 0.03, 1, 0.95])