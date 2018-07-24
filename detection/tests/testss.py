# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:40:54 2018

@author: aag14
"""

for imageID, imageMeta in genVal.imagesMeta.items():
    rels = np.array(imageMeta['rels'])
    if 2 in rels[:,2]:
        print(imageID)