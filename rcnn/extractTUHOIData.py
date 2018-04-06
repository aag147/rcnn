# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:30 2018

@author: aag14
"""
import plotData as pdata
import csv
url = 'C:/Users/aag14/Documents/Skole/Speciale/TUHOI_instruments/'
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


def getURL():
    return url

def getUniqueLabels():
    labels = np.array([['play'+i, 'hold'+i] for i in instruments if i not in nonactive_instruments])
    labels = labels.reshape([2*8])
    #labels = np.append(labels, 'other')
    
    labels = dict(zip(iter(labels), range(len(labels))))
    return labels

def getLabelStats(imagesMeta):
    stats = {i:{'play':0, 'hold': 0} for i in instruments}
    stats['total'] = 0
    for imageID, imageMeta in imagesMeta.items():
        for relID, rel in imageMeta['rels'].items():    
            stats[rel['obj']][rel['pred']] += 1
            stats['total'] += 1
    return stats

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
    i = 0
    newImagesMeta = {}
    rawImagesMeta = {}
    
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
           garbage = []
           
           for i in range(len(imageObjs)):
               objName     = imageObjs[i]
               preds       = imagePreds[i].split("\n")
               
               pred = None
               votesPlay = sum(predsPlay.count(i.lower()) for i in preds)
               votesHold = sum(predsHold.count(i.lower()) for i in preds)
               if votesHold >= votesPlay:
                   pred = 'hold'
               else:
                   if votesPlay > 0:
                       pred = 'play'
                       
               if objName in ['acoustic guitar']:
                   objName = 'guitar'
                   print('acustic')
               
               if objName in instruments:
                   if pred in ['play', 'hold']:
                       rels[objName] = pred
                   else:
                       print(imageID + ": " + "" + " " + objName)
                       print(preds)
               else:
                   if objName:
                       garbage.append(objName)
           if rels:
               #if garbage:
               #    print(garbage)
               imageMeta['rels'] = rels
               newImagesMeta[imageID] = imageMeta
               rawImagesMeta[imageID] = line
           #if len(rels) > 1:
               #print(line[19:30])
               #print(imageID)

        i += 1
    return rawImagesMeta, newImagesMeta

def getBoundingBoxes(imagesMeta, objects, labels):
    newImagesMeta = {}
    imagesBadOnes = {}
    noImage = 0

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
                    relsTmp.append({'label': label, 'pred': pred, 'obj': objName, 'objBB': bb})
                else:
                    relsObjBad.append({'pred': pred, 'name': objName, 'bb': bb})
        
        # Choose best person boxes
        bestPrs = np.array([[0.0, 0.0, None] for i in range(len(relsTmp))])
        prsIdx = 0
        for perBB in persons:
            IoUs = np.zeros([len(relsTmp),2])
            relIdx = 0
            for rel in relsTmp:
                objBB = rel['objBB']
                IoUPsy = pdata.get_iou(objBB, perBB, False)
                IoU = pdata.get_iou(objBB, perBB)
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
    return newImagesMeta, imagesBadOnes


if __name__ == "__main__":
    plt.close("all")
    #deleteFillerFiles(images, 'bbox', 'XML')
    
    rawImagesMeta, imagesMeta = extractMetaData()
    objects, allObjects = extractObjectData()
    imagesMeta, samples, imagesBadOnes = getBoundingBoxes(imagesMeta, objects)
    
    images = list(imagesMeta.keys())
    images.sort()
    
    pdata.drawImages(images[383:387], imagesMeta, url+'images/', imagesBadOnes)
