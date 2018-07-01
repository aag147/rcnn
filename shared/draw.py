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
import itertools
import math


def plot_hoi_stats(stats, sort=False):
    f, spl = plt.subplots(1)
    names = []
    idxs = []
    counts = []
    for obj, preds in stats.items():
        if type(preds) is dict:
            for pred, count in preds.items():
                if pred not in ['nb_images', 'nb_samples', 'total', 'totalx']:
                    names.append(pred)
                    idxs.append(len(idxs))
                    counts.append(count)
    if sort:
        counts.sort()
    spl.bar(idxs, counts[::-1], bottom=0.1)
#    spl.set_xlim([0, len(counts)])
    spl.set_xticks([])
    spl.set_yscale('log')
    spl.axis((-1,len(counts),0.1,10**5.5))
    
    xlabel = 'Classes'
    if sort:
        xlabel = 'Sorted classes'
    spl.set_ylabel('Count')
    spl.set_xlabel(xlabel)

def plot_object_stats(stats, sort=False):
    f, spl = plt.subplots(1)
    names = []
    idxs = []
    counts = []
    for key, count in stats.items():
        if key not in ['nb_images', 'nb_samples', 'total', 'totalx', 'bg']:
            names.append(key)
            idxs.append(len(idxs))
            counts.append(count)
    if sort:
        counts.sort()
    spl.bar(idxs, counts, bottom=0.01)
    spl.set_yscale('log')
    spl.set_xticks([])
    spl.set_xticklabels(names)
    spl.set_ylabel('Count')
    spl.axis((-1,len(counts),0.1,10**5.5))
    
    xlabel = 'Classes'
    if sort:
        xlabel = 'Sorted classes'
    spl.set_xlabel(xlabel)
    print(names)
    
def plot_area_stats(stats, sort=False):
    f, spl = plt.subplots(1)
    
    names = []
    idxs = []
    counts = []
    for key, count in stats.items():
        names.append(key)
        idxs.append(len(idxs))
        counts.append(count)
    
    counts = np.array(counts)
    names = np.array(names)
    sort_idxs = np.argsort(names)
    counts = counts[sort_idxs]
    names = list(names[sort_idxs][0:11])
    print(counts)
    print(names)
    spl.bar(idxs, counts, bottom=1)
#    spl.plot(idxs, counts)
    spl.set_yscale('log')
#    spl.set_xticks([])
    plt.xticks(np.arange(0, 11, 1))
    spl.set_xticklabels(names)
    spl.set_ylabel('Count')
    spl.set_xlabel('Nearest anchor size')
    spl.axis((-1,len(counts),10**0,10**5.5))


def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)

#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def plotLosses(log):
    # summarize history for loss
    f, spl = plt.subplots(1)
    spl = spl.ravel()
    plt.plot(log.history['loss'])
    plt.plot(log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def drawHoIComplete(img, h_bbox, o_bbox, pattern, labels, label_mapping, cfg):
    f, spl = plt.subplots(2,2)
    spl = spl.ravel()    
    
    hbox = drawProposalBox(h_bbox * cfg.rpn_stride)
    obox = drawProposalBox(o_bbox * cfg.rpn_stride)
    spl[0].imshow(img)
    spl[0].plot(hbox[0,:], hbox[1,:])
    spl[0].plot(obox[0,:], obox[1,:])
    
    spl[2].imshow(pattern[:,:,0])
    spl[3].imshow(pattern[:,:,1])
    
    lbs = ', '.join([label_mapping[x]['pred_ing'] for x in np.where(labels==1)[0]]) + ' ' + label_mapping[np.where(labels==1)[0][0]]['obj'] if np.sum(labels)>0 else 'none'
    
    print('label:', lbs)
    
def drawPositiveHoIs(img, h_bbox, o_bbox, labels, label_mapping, imageMeta, imageDims, cfg):
    import filters_helper as helper
    ps_idxs = np.where(np.sum(labels>0.5, axis=1)>0)[0]
    nb_imgs = len(ps_idxs)
    
    f, spl = plt.subplots(math.ceil((nb_imgs+1)/2), 2)
    spl = spl.ravel()    
    
    # GT Boxes
    gt_boxes = helper.normalizeGTboxes(imageMeta['objects'], scale=imageDims['scale'], roundoff=True)
    for bb in gt_boxes:
        bbox = drawBoundingBox(bb)
        spl[0].imshow(img)
        spl[0].plot(bbox[0,:], bbox[1,:])

    spl_idx = 1
#    print(ps_idxs)
    for i in range(nb_imgs): 
        idx = ps_idxs[i]
        hbox = drawProposalBox(h_bbox[idx,:] * cfg.rpn_stride)
        obox = drawProposalBox(o_bbox[idx,:] * cfg.rpn_stride)
        spl[spl_idx].imshow(img)
        spl[spl_idx].plot(hbox[0,:], hbox[1,:])
        spl[spl_idx].plot(obox[0,:], obox[1,:])
#        print(labels.shape)
        lbs = ', '.join([label_mapping[x]['pred_ing'] for x in np.where(labels[idx,:]>0.5)[0]]) + ' ' + label_mapping[np.where(labels[idx,:]>0.5)[0][0]]['obj'] if np.sum(labels[idx,:])>0 else 'none'
        print('label:', lbs)
        spl_idx += 1
    


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
        

def drawAnchors(img, anchorsGT, cfg):
    f, spl = plt.subplots(1)
    spl.imshow(img)
    bboxes = []
    for anchor in anchorsGT:
        objectiveness = anchor[4]
        if objectiveness>0.5:
            c = 'red'
        else:
            continue
            c = 'blue'
        bb = anchor[0:4]*cfg.rpn_stride
        bbox = drawProposalBox(bb)
        spl.plot(bbox[0,:], bbox[1,:], c=c)
        bboxes.append(bb)
    return np.array(bboxes)
        
def drawPositiveAnchors(img, anchorsGT, cfg):
    f, spl = plt.subplots(1)
    spl.axis('off')
    spl.imshow(img)
    bboxes = []
    for anchor in anchorsGT:
        objectiveness = anchor[4]
        if objectiveness>0.9:
            bb = anchor[0:4]*cfg.rpn_stride
            bbox = drawProposalBox(bb)
            spl.plot(bbox[0,:], bbox[1,:])
            bboxes.append(bb)
    return np.array(bboxes)

def drawOverlapAnchors(img, anchors, imageMeta, imageDims, cfg):
    import filters_helper as helper
    import utils
    f, spl = plt.subplots(1)
#    spl.axis('off')
    spl.imshow(img)
    bboxes = []
    gta = helper.normalizeGTboxes(imageMeta['objects'], scale=imageDims['scale'], rpn_stride=cfg.rpn_stride)
    
    for anchor in anchors:
        (xmin, ymin, width, height) = anchor[0:4]       
        rt = {'xmin': xmin, 'ymin': ymin, 'xmax': xmin+width, 'ymax': ymin+height}
        best_iou = 0.0
        for bbidx, gt in enumerate(gta):
            curr_iou = utils.get_iou(gt, rt)
            if curr_iou > best_iou:
#                print(curr_iou)
                best_iou = curr_iou
        if best_iou>=0.5:
            bb = {key:x*cfg.rpn_stride for key,x in rt.items()}
            bbox = drawBoundingBox(bb)
            spl.plot(bbox[0,:], bbox[1,:])
            bboxes.append(bb)
        
#    for bbidx, gt in enumerate(gta):
#        bb = {key:x*cfg.rpn_stride for key,x in gt.items()}
#        bbox = drawBoundingBox(bb)
#        spl.plot(bbox[0,:], bbox[1,:])
    return np.array(bboxes)

def drawOverlapRois(img, rois, imageMeta, imageDims, cfg, obj_mapping):
    import filters_helper as helper
    import utils
    f, spl = plt.subplots(1)
#    spl.axis('off')
    spl.imshow(img)
    bboxes = []
    gta = helper.normalizeGTboxes(imageMeta['objects'], scale=imageDims['scale'], rpn_stride=cfg.rpn_stride)
    inv_obj_mapping = {x:key for key,x in obj_mapping.items()}
    for roi in rois:
        (xmin, ymin, width, height) = roi[0:4]   
        label = int(roi[5])
        prop = roi[4]
        rt = {'xmin': xmin, 'ymin': ymin, 'xmax': xmin+width, 'ymax': ymin+height}
        best_iou = 0.0
        for bbidx, gt in enumerate(gta):
            gt_label = obj_mapping[gt['label']]
            if label != gt_label:
                continue
            curr_iou = utils.get_iou(gt, rt)
            if curr_iou > best_iou:
#                print(curr_iou)
                best_iou = curr_iou
        if best_iou>=0.5:
            c = 'red'
            print('Pos. label:', inv_obj_mapping[label], prop, best_iou)
        elif best_iou>=0:
            c = 'blue'
            print('Neg. label:', inv_obj_mapping[label], prop, best_iou)
        else:
            continue
        bb = {key:x*cfg.rpn_stride for key,x in rt.items()}
        bbox = drawBoundingBox(bb)
        spl.plot(bbox[0,:], bbox[1,:], c=c)
        bboxes.append(bb)
        
#    for bbidx, gt in enumerate(gta):
#        bb = {key:x*cfg.rpn_stride for key,x in gt.items()}
#        bbox = drawBoundingBox(bb)
#        spl.plot(bbox[0,:], bbox[1,:])
    return np.array(bboxes)

def drawPositiveRois(img, rois, obj_mapping):
    f, spl = plt.subplots(1)
    spl.imshow(img)
    bboxes = []
    for roi in rois:
        labelID = int(roi[5])
        if labelID>0:
            bb = roi[0:4]*16
            bbox = drawProposalBox(bb)
            spl.plot(bbox[0,:], bbox[1,:])
            bboxes.append(bb)
    return np.array(bboxes)


def drawBoxes(img, bboxes, imageDims):
    f, spl = plt.subplots(1)
    spl.imshow(img)
    scales = imageDims['scale']
    for bbox in bboxes:
        bbox = drawBoundingBox(bbox)
        print(bbox[1,:]*scales[0])
        spl.plot(bbox[0,:]*scales[1], bbox[1,:]*scales[0])
            
def drawGTBoxes(img, imageMeta, imageDims):
    import filters_helper as helper
    f, spl = plt.subplots(1)
    spl.axis('off')
    spl.imshow(img)
    
    bboxes = helper.normalizeGTboxes(imageMeta['objects'], scale=imageDims['scale'], roundoff=True)
    
    for bb in bboxes:
        bbox = drawBoundingBox(bb)
        spl.plot(bbox[0,:], bbox[1,:])
    return bboxes

def drawBoundingBox(bb):
    xmin = bb['xmin']; xmax = bb['xmax']
    ymin = bb['ymin']; ymax = bb['ymax']
    
    box = np.array([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]])
    return box

def drawProposalBox(bb):    
    xmin = bb[0]; xmax = bb[1]
    ymin = bb[2]; ymax = bb[3]
    
    xmin = bb[0]; ymin = bb[1]
    xmax = bb[2]; ymax = bb[3]

    xmin = bb[0]; xmax = xmin + bb[2]
    ymin = bb[1]; ymax = ymin + bb[3]
    
    box = np.array([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]])
    return box


def drawHOI(image, prsBB, objBB):
    colours = [['#FE5757', '#e44e4e'], ['#57FE57', '#4ee44e'], ['#5757FE', '#4e4ee4'], ['#AB57FE', '#994ee4']]
    c = colours[0]
    f, spl = plt.subplots(1,2)
    spl = spl.ravel()
    
#    image = image.transpose([1,2,0])
    spl[0].imshow(image)
    
    personBB = drawProposalBox(prsBB)
    objectBB = drawProposalBox(objBB)
    spl[0].plot(personBB[0,:], personBB[1,:], c=c[0])
    spl[1].plot(objectBB[0,:], objectBB[1,:], c=c[1])

def drawCropxResize(image, prsBB, objBB):
    colours = [['#FE5757', '#e44e4e'], ['#57FE57', '#4ee44e'], ['#5757FE', '#4e4ee4'], ['#AB57FE', '#994ee4']]
    c = colours[0]
    f, spl = plt.subplots(1,2)
    spl = spl.ravel()
    
    image = image.transpose([1,2,0])
    spl[0].imshow(image)
    
    personBB = drawProposalBox(prsBB)
    objectBB = drawProposalBox(objBB)
    spl[0].plot(personBB[0,:], personBB[1,:], c=c[0])
    spl[0].plot(objectBB[0,:], objectBB[1,:], c=c[1])


def drawHoICrops(prsCrop, objCrop, patterns):
    f, spl = plt.subplots(2,2)
    spl = spl.ravel()
    
    spl[0].imshow(prsCrop)
    spl[1].imshow(objCrop)
    spl[2].imshow(patterns[:,:,0])
    spl[3].imshow(patterns[:,:,1])
    
    
def drawObjExample(imageMeta, images_path):
    colours = ['#FEc75c', '#123456','#456789', '#abcdef','#fedcba', '#987654','#654321', '#994ee4']
    
    img = cv.imread(images_path + imageMeta['imageName'])
    assert img is not None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    objs = []
    names = []
    for obj in imageMeta['objects']:
        label = obj['label']
        objBB = drawBoundingBox(obj)
        objs.append(objBB)
        names.append(label)
        
    f, spl = plt.subplots(1,1)
    spl.axis('off')
    spl.imshow(img)
    for j, obj in enumerate(objs):
        spl.plot(obj[0,:], obj[1,:], c=colours[j])
#        f.text(0.55,0.95,names[j], ha="center", va="bottom", size="medium",color=colours[j])
        print(names[j], colours[j])
        
        


def drawHoIExample(imageMeta, images_path, hoi_mapping):    
    img = cv.imread(images_path + imageMeta['imageName'])
    assert img is not None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    all_objs = imageMeta['objects']
    
    objs = []
    prss  = []
    lines = []
    names = []
    for rel in imageMeta['rels']:
        label = hoi_mapping[rel[2]]
        obj = all_objs[rel[0]]
        prs = all_objs[rel[1]]
        objBB = drawBoundingBox(obj)
        prsBB = drawBoundingBox(prs)
        
        prsC = [prs['ymin']+(prs['ymax']-prs['ymin'])/2, prs['xmin']+(prs['xmax']-prs['xmin'])/2]
        objC = [obj['ymin']+(obj['ymax']-obj['ymin'])/2, obj['xmin']+(obj['xmax']-obj['xmin'])/2]
        line = np.array([prsC, objC])
        objs.append(objBB)
        prss.append(prsBB)
        lines.append(line)
        names.append(label)
        
    for j, _ in enumerate(imageMeta['rels']):
        f, spl = plt.subplots(1,1)
        obj = objs[j]
        prs = prss[j]
        line = lines[j]
        spl.axis('off')
        spl.imshow(img)
        spl.plot(obj[0,:], obj[1,:], c='green')
        spl.plot(prs[0,:], prs[1,:], c='blue')
        spl.plot(line[:,1], line[:,0], c='red')
        spl.scatter(line[:,1], line[:,0], c='red', s=5)
        print(names[j])
        if j == 2:
            break

def drawImages(imagesID, imagesMeta, labels, path, imagesBadOnes = False):
    f, spl = plt.subplots(2,2)
    spl = spl.ravel()
    i = 0
    for imageID in imagesID:
        print(imageID)
        imageMeta = imagesMeta[imageID]
        image = cv.imread(path + imageMeta['imageName'])
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