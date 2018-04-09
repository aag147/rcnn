# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 15:15:26 2018

@author: aag14
"""

newImagesMeta = {}

for imageID, imageMeta in imagesMeta.items():
    newImageMeta = {'ID': imageMeta['ID'], 'imageID': imageMeta['imageID']}
    newRels = {}
    for relID, rel in imageMeta['rels'].items():
        newRel = {'labels': [rel['label']], 'names': [{'pred': rel['pred'], 'obj': rel['obj']}], 'prsBB': rel['prsBB'], 'objBB': rel['objBB'], 'prsID': rel['prsID']}
        newRels[relID] = newRel
    newImageMeta['rels'] = newRels
    newImagesMeta[imageID] = newImageMeta