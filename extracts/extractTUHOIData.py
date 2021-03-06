# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:30 2018

@author: aag14
"""
import utils, draw
import csv
instruments = ["bassoon", "cello", "clarinet", "erhu", "flute", "french horn", "guitar", "harp", "recorder", "saxophone", "trumpet", "violin"]

nonactive_instruments = ['bassoon', 'clarinet', 'erhu', 'recorder']

predsPlay = ['play', 'blow', 'touch', 'strum', 'struck']
predsHold = ['hold']

objInx = [18, 21, 23, 25, 26, 27, 28, 29, 30]
predInx = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


import glob
import os
import cv2 as cv
import xml.etree.ElementTree as ET  
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np

def getUniqueLabels(cfg):
    labels = np.array([[{'pred':'play', 'pred_ing':'playing', 'obj':i}, {'pred':'hold', 'pred_ing': 'holding', 'obj':i}] for i in instruments if i not in nonactive_instruments])
    labels = labels.reshape([2*8])
    labels = list(labels)
    #labels = np.append(labels, 'other')
    
#    labels = dict(zip(iter(labels), range(len(labels))))
    return labels

def extractObjectData():
    objectData = sio.loadmat(url + 'meta_det.mat', struct_as_record=False, squeeze_me=True)['synsets']
    
    objects = {}
    allObjects = {}
    for elem in objectData:
        objID   = elem.WNID
        objName = elem.name
        if objName in instruments or objName == 'person':
            objects[objID] = objName
        allObjects[objID] = objName
    return objects, allObjects

def extractMetaData():
    j = 0
    newImagesMeta = {}
    rawImagesMeta = {}
    garbage = []
    
    with open(url + 'crowdflower_result.csv', newline='') as csvfile:
        f = list(csv.reader(csvfile))
        #newImagesMeta['Alegend'] = imagesMeta[0]
        for line in f:
           line = np.array(line)
           relID   = line[0]
           imageID = line[17]
           
           imageObjs = line[objInx]
           imagePreds = line[predInx]
           
           imageMeta = {'ID': relID, 'imageID': imageID + '.JPEG'}
           rels = {}
           for i in range(len(imageObjs)):
               objName     = imageObjs[i]
               preds       = imagePreds[i].split("\n")
               
               # Objects
               if objName in ['acoustic guitar']:
                   objName = 'guitar'
                   print('acustic')
               
               if objName not in instruments:
                   garbage.append(objName)
                   continue
               
               #Predicates
               pred = None
               votesPlay = sum(predsPlay.count(i.lower()) for i in preds)
               votesHold = sum(predsHold.count(i.lower()) for i in preds)
               if votesHold >= votesPlay:
                   pred = 'hold'
               elif votesPlay > 0:
                   pred = 'play'
#               else:
#                   pred = hold
               
               if pred not in ['play', 'hold']:
                   print(imageID + ": " + "" + " " + objName)
                   print(preds)
                   continue
               
               rels[objName] = pred

           if rels:
               #if garbage:
               #    print(garbage)
               imageMeta['rels'] = rels
               newImagesMeta[imageID] = imageMeta
               rawImagesMeta[imageID] = line
           #if len(rels) > 1:
               #print(line[19:30])
               #print(imageID)

           j += 1
    garbage = set(garbage)
    print(len(f))
    return newImagesMeta, rawImagesMeta, garbage

def getBoundingBoxes(imagesMeta, objects, labels):
    newImagesMeta = {}
    imagesBadOnes = {}
    noImage = 0
    total = 0

    for imageID, imageMeta in imagesMeta.items():
        try:
            root = ET.parse(url + 'bbox/' + imageID + '.xml').getroot()
        except FileNotFoundError as e:
            print("missing",imageID)
            continue
        relsObj = imageMeta['rels']
        relsTmp = []
        relsObjBad = []
        relsPrsBad = []
        persons = []
        
        # Add objects
        for elem in root:
            if elem.tag != "object":
                continue
            
            # BB name
            objID = elem.find('name').text
            if objID not in objects.keys():
                continue
            objName = objects[objID]
            
            ## BB coordinates
            bbXML   = elem.find('bndbox')
            xmin = int(bbXML.find('xmin').text)
            xmax = int(bbXML.find('xmax').text)
            ymin = int(bbXML.find('ymin').text)
            ymax = int(bbXML.find('ymax').text)
            bb = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax} 
            
            if objName == 'person':
                persons.append(bb)
            else:
                #Meta relation
                if objName in relsObj.keys():
                    pred = relsObj[objName]
                    label = labels[pred+objName]
                    relsTmp.append({'labels': [label], 'names': [{'pred': pred, 'obj': objName}], 'objBB': bb})
                else:
                    relsObjBad.append({'pred': pred, 'name': objName, 'bb': bb})
                total += 1
        
        # Choose best person boxes
        bestPrs = np.array([[0.0, 0.0, None] for i in range(len(relsTmp))])
        prsIdx = 0
        for perBB in persons:
            IoUs = np.zeros([len(relsTmp),2])
            relIdx = 0
            for rel in relsTmp:
                objBB = rel['objBB']
                IoUPsy = utils.get_iou(objBB, perBB, False)
                IoU = utils.get_iou(objBB, perBB)
                IoUs[relIdx,:] = [IoUPsy, IoU]
                relIdx += 1 
            bestIdx = np.argmax(IoUs[:,0])
            if IoUs[bestIdx,0] > bestPrs[bestIdx,0] or IoUs[bestIdx,0] == bestPrs[bestIdx,0] and IoUs[bestIdx,1] > bestPrs[bestIdx,1]:
                bestPrs[bestIdx,:] = np.array([IoUs[bestIdx,0], IoUs[bestIdx,1], prsIdx])
            prsIdx += 1

        # Add best persons
        relsFinal = {}        
        relIdx = 0
        objGood = False
        perGood = True
        for [bestIoUPsy, _, prsIdx] in bestPrs:
            if bestIoUPsy > 0.1:
                relTmp = relsTmp[relIdx]
                relTmp['prsBB'] = persons[int(prsIdx)]
                relTmp['prsID'] = int(prsIdx)
                relsFinal[relIdx] = relTmp
            else:
                perGood = False
            relIdx += 1
            objGood = True
            
        # Add bad persons
        bestPrsIdx = bestPrs[:,2]
        bestPrsIdx = bestPrsIdx[bestPrsIdx != np.array(None)]
        bestPrsIdx = bestPrsIdx.astype(int)
        for i in range(len(persons)):
            if i not in bestPrsIdx:
                relsPrsBad.append({'pred': '', 'name': 'person', 'bb': persons[i]})
            
        if not objGood or not perGood:
            continue

        imageMeta['rels'] = relsFinal
        newImagesMeta[imageID] = imageMeta
        imagesBadOnes[imageID] = [relsObjBad, relsPrsBad]
        noImage += 1
    #print(badOnes)
    print(total)
    return newImagesMeta, imagesBadOnes


if __name__ == "__main__":
    plt.close("all")
    #deleteFillerFiles(images, 'bbox', 'XML')
    
    imagesMeta, rawImagesMeta, garbage = extractMetaData()
    objects, allObjects = extractObjectData()
    unique_labels = getUniqueLabels()
    imagesMeta, imagesBadOnes = getBoundingBoxes(imagesMeta, objects, unique_labels)
    imagesMeta = utils.load_dict('TU_PPMI', url)
    images = list(imagesMeta.keys())
    images.sort()
    
    stat = getLabelStats(imagesMeta)
    
#    draw.drawImages(images[383:387], imagesMeta, url+'images/', imagesBadOnes)
