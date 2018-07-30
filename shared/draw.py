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
                          no_bg=False,
                          name='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.copy(cm)
    if normalize:
        if no_bg:
            cm = cm[1:,1:]
            cm = cm.astype('float') / (cm.sum(axis=0)[np.newaxis,:] + 0.000000000000001)
        else:
            cm[:,1:] = cm[:,1:].astype('float') / (cm[:,1:].sum(axis=0)[np.newaxis,:] + 0.000000000000001)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    cm_padding = np.zeros([x+10*2 for x in cm.shape])
    cm_padding[10:-10,10:-10] = cm
#    print(cm)
    f = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name)
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
    return cm

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
    
    
def plotRPNLosses(hist, mode='rpn', yaxis=None):
    x = hist[:,0]
    t = hist[:,1]
    tc = hist[:,3]
    tr = hist[:,4]
    v = hist[:,5]
    vc = hist[:,7]
    vr = hist[:,8]
    
    f, spl = plt.subplots(1)
    spl.plot(x, t, c=(0,0,1))
    
    if mode in ['rpn','det']:
        spl.plot(x, tc, c=(0.5,0,1.0))
        spl.plot(x, tr, c=(0.5,0,0.8))
        
    spl.plot(x, v, c=(0,1,0))
    if mode in ['rpn','det']:
        spl.plot(x, vc, c=(0.5,1.0,0.0))
        spl.plot(x, vr, c=(0.5,0.8,0.0))
        
    if mode in ['rpn', 'det']:
        nb_itr = 49
    else:
        nb_itr = 30
    
#    plt.title('DET loss')
    if yaxis is not None:
        spl.set_yscale('log')
        spl.axis((-1,nb_itr+1,10**-1,10**1.7))
    plt.ylabel('Loss')
    plt.xlabel('Iteration (in 10K)')
    if mode == 'hoi':
        plt.legend(['train+L2', 'test+L2'], loc='upper right')
    else:
        plt.legend(['train+L2', 'train cls', 'train regr', 'val+L2', 'val cls', 'val regr'], loc='upper right')
    plt.show()    
    
    
def plotFasterLosses(hists, mode='rpn', yaxis=None):
    [hist1, hist2, hist3] = hists
    x = hist1[:,0]
    I = hist1[:,5]
    Ic = hist1[:,7]
    Ir = hist1[:,8]
    II = hist2[:,5]
    IIc = hist2[:,7]
    IIr = hist2[:,8]
    III = hist3[:,5]
    IIIc = hist3[:,7]
    IIIr = hist3[:,8]
    
    f, spl = plt.subplots(1)
    

    if mode in ['rpn','det']:
        spl.plot(hist1[:,0], Ic, c=(0,0.5,1.0))
        spl.plot(hist1[:,0], Ir, c=(0,0.5,0.8))
    else:
        spl.plot(x, I, c=(0,0,1))
        

    if mode in ['rpn','det']:
        spl.plot(hist2[:,0], IIc, c=(0.5,1.0,0.0))
        spl.plot(hist2[:,0], IIr, c=(0.5,0.8,0.0))
    else:
        spl.plot(x, II, c=(0,1,0))
        
    if mode in ['rpn','det']:
        spl.plot(hist3[:,0], IIIc, c=(1.0,0.0,0.5))
        spl.plot(hist3[:,0], IIIr, c=(0.8,0.0,0.5))
    else:
        spl.plot(x, III, c=(1,0,0))
        
    if mode in ['rpn', 'det']:
        nb_itr = 49
    else:
        nb_itr = 30
    
#    plt.title('DET loss')
    if yaxis is not None:
        spl.set_yscale('log')
        spl.axis((-1,nb_itr+1,10**-1,10**0.5))
    plt.ylabel('Loss')
    plt.xlabel('Iteration (in 10K)')
    if mode == 'hoi':
        plt.legend(['Model I+L2', 'Model II+L2', 'Model III+L2'], loc='upper right')
    else:
        plt.legend(['Model I cls', 'Model I regr', 'Model II cls', 'Model II regr', 'Model III cls', 'Model III regr'], loc='upper right')
    plt.show()    

def pltAPs(APs, yaxis=None):
    x = list(range(len(APs)))
    f, spl = plt.subplots(1)
    spl.bar(x, APs, bottom=0.00000000000000000001)

    if yaxis is not None:
        spl.set_yscale('log')
        spl.axis((-1,len(APs),10**-2,10**0))
    else:
        spl.axis((-1,len(APs),0,0.7))
    plt.ylabel('AP')
    plt.xlabel('Classes')
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
    
#    f, spl = plt.subplots(math.ceil((nb_imgs+1)/2), 2)
    f, spl = plt.subplots(3, 2)
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
        if spl_idx == 6:
            break
    


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
    spl.axis('off')
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
    f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
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
    spl.axis('off')
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
            c = 'red'
        else:
            c = 'blue'
            continue
        bb = {key:x*cfg.rpn_stride for key,x in rt.items()}
        bbox = drawBoundingBox(bb)
        spl.plot(bbox[0,:], bbox[1,:], c=c)
        bboxes.append(bb)
        
    f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        
#    for bbidx, gt in enumerate(gta):
#        bb = {key:x*cfg.rpn_stride for key,x in gt.items()}
#        bbox = drawBoundingBox(bb)
#        spl.plot(bbox[0,:], bbox[1,:])
    return (bboxes)

def drawOverlapRois(img, rois, imageMeta, imageDims, cfg, obj_mapping):
    import filters_helper as helper
    import utils
    f, spl = plt.subplots(1)
    spl.axis('off')
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
            continue
            print('Neg. label:', inv_obj_mapping[label], prop, best_iou)
        else:
            continue
        bb = {key:x*cfg.rpn_stride for key,x in rt.items()}
        bbox = np.copy(drawBoundingBox(bb))
        spl.plot(bbox[0,:], bbox[1,:], c=c)
        bboxes.append(bb)
        
    f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
#    for bbidx, gt in enumerate(gta):
#        bb = {key:x*cfg.rpn_stride for key,x in gt.items()}
#        bbox = drawBoundingBox(bb)
#        spl.plot(bbox[0,:], bbox[1,:])
    return (bboxes)


def drawHumanAndObjectRois(img, rois, imageMeta, obj_mapping):
    f, spl = plt.subplots(2,1)
    spl[0].axis('off')
    spl[1].axis('off')
    spl[0].imshow(img)
    spl[1].imshow(img)
    
    gt_labels = [obj_mapping[x['label']] for x in imageMeta['objects']]
    
    bboxes = []
    hi = 0
    oi = 0
    for roi in rois:
        labelID = int(roi[5])
        if labelID==1 and hi < 15:
            bb = roi[0:4]*16
            bbox = drawProposalBox(bb)
            spl[0].plot(bbox[0,:], bbox[1,:], c='red')
            bboxes.append(bb)
            hi += 1
        elif labelID > 1 and oi < 15 and labelID in gt_labels:
            bb = roi[0:4]*16
            bbox = drawProposalBox(bb)
            spl[1].plot(bbox[0,:], bbox[1,:], c='blue')
            bboxes.append(bb)
            oi += 1

def drawPositiveRois(img, rois, obj_mapping):
    f, spl = plt.subplots(1)
    spl.axis('off')
    spl.imshow(img)
    bboxes = []
    for roi in rois:
        labelID = int(roi[5])
        if labelID==9:
            bb = roi[0:4]*16
            bbox = drawProposalBox(bb)
            spl.plot(bbox[0,:], bbox[1,:], c='red')
            bboxes.append(bb)
    f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
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
        spl.plot(bbox[0,:], bbox[1,:], c='red')
        
    f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
#    f.tight_layout()    
    return bboxes

def drawPositiveHoI(img, hbboxes, obboxes, patterns, props, imageMeta, imageDims, cfg, obj_mapping):
#    f, spl = plt.subplots(2,2)
#    spl = spl.ravel()
#    spl[0].imshow(img)
    inv_obj_mapping = {x:key for key,x in obj_mapping.items()}
    colours = ['#FEc75c', '#123456','#456789', '#abcdef','#fedcba', '#987654','#654321', '#994ee4']
    idxs = []
    nb_pairs = hbboxes.shape[0]
    c_idx=0
    for idx in range(nb_pairs):
#        if idx > 5 and idx < 100:
#            continue
#        if idx > 105:
#            break
#        hprop = (hbboxes[idx,4])
#        oprop = (obboxes[idx,4])
#        hlbl = int(hbboxes[idx,5])
#        olbl = int(obboxes[idx,5])
        hoilabel = np.where(props[idx,:]>0.001)[0]
        if len(hoilabel)>0 or True:
            hoiprop = props[idx,hoilabel]
#            print(idx, hoilabel, hoiprop)
            
            f, spl = plt.subplots(2,2)
            spl = spl.ravel()   
            spl[0].imshow(img)
            c = colours[c_idx]
            hbbox = hbboxes[idx,:4]*16
            obbox = obboxes[idx,:4]*16
            hbbox = drawProposalBox(hbbox)
            obbox = drawProposalBox(obbox)
            spl[0].plot(hbbox[0,:], hbbox[1,:], c=c)
            spl[0].plot(obbox[0,:], obbox[1,:], c=c)
            if patterns is not None:
                spl[2].imshow(patterns[idx,:,:,0])
                spl[3].imshow(patterns[idx,:,:,1])
            idxs.append(idx)
#            print('Pos. label:', inv_obj_mapping[hlbl], inv_obj_mapping[olbl], hprop, oprop, hoiprop)
            
            c_idx = (c_idx+1) % len(colours)
    return np.array(idxs)


def drawOverlapHOIRes(evalData, imagesMeta, obj_mapping, hoi_mapping, images_path):

    prev_imageID = ''
    
    for idx, line in enumerate(evalData):
        line = cp.copy(line)
        imageID = line['image_id']
        if imageID != prev_imageID:
            imageMeta = imagesMeta[imageID]
            img = cv.imread(images_path + imageMeta['imageName'])
            assert img is not None
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            import filters_helper as helper    
            gt_hbboxes, gt_obboxes = helper._transformGTBBox(imageMeta['objects'], obj_mapping, np.array(imageMeta['rels']), dosplit=True)
            gt_rels = helper._getRealRels(np.array(imageMeta['rels']))
            gt_objs = np.unique(gt_obboxes[:,4])
        
        
        hbbox = line['hbbox']
        obbox = line['obbox']
        label = line['category_id']
        prop  = line['score']
        
        overlap = 0
        for gt_idx, gt_rel in enumerate(gt_rels):
            gt_hbbox = gt_hbboxes[gt_rel[0]:gt_rel[0]+1]
            gt_obbox = gt_obboxes[gt_rel[1]:gt_rel[1]+1]
            gt_label = gt_rel[2]
            
            if gt_label != label:
                continue
                        
            hum_overlap = helper._computeIoUs(hbbox, gt_hbbox)[0]
            obj_overlap = helper._computeIoUs(obbox, gt_obbox)[0]
            
            if min(hum_overlap, obj_overlap) > overlap:
                overlap = min(hum_overlap, obj_overlap)
    
        hbbox = np.copy(hbbox)
        obbox = np.copy(obbox)
        hbbox[2] -= hbbox[0]
        hbbox[3] -= hbbox[1]
        obbox[2] -= obbox[0]
        obbox[3] -= obbox[1]


        hdotx = hbbox[0] + (hbbox[2] / 2)
        hdoty = hbbox[1] + (hbbox[3] / 2)
        odotx = obbox[0] + (obbox[2] / 2)
        odoty = obbox[1] + (obbox[3] / 2)
        conn = np.array([[odoty, odotx], [hdoty, hdotx]])
        hbbox = drawProposalBox(hbbox)
        obbox = drawProposalBox(obbox)

        titlename = hoi_mapping[label]['pred_ing'] + ' ' + hoi_mapping[label]['obj']
        titlename += '(%.2f)' % prop

        if overlap >= 0.5:
            
            f, spl = plt.subplots(1,1)
            spl.axis('off')
            spl.imshow(img)
            
            spl.plot(hbbox[0,:], hbbox[1,:], c='green')
            spl.plot(obbox[0,:], obbox[1,:], c='blue')
            spl.plot(conn[:,1], conn[:,0], c='red')
            spl.scatter(conn[:,1], conn[:,0], c='red', s=5)
            f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
#            f.suptitle('')
            
            print(str(idx) + '. Pos. label:', hoi_mapping[label], prop, overlap, imageID) 
        else:
#            continue
            f, spl = plt.subplots(1,1)
            spl.axis('off')
            spl.imshow(img)

            spl.plot(hbbox[0,:], hbbox[1,:], c='green')
            spl.plot(obbox[0,:], obbox[1,:], c='blue')
            spl.plot(conn[:,1], conn[:,0], c='red')
            spl.scatter(conn[:,1], conn[:,0], c='red', s=5)
            f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            f.suptitle('Neg: ' + titlename)
            print('Neg. label:', hoi_mapping[label], prop, line['eval'], overlap, imageID)
#            f, spl = plt.subplots(1,1)
#            spl.axis('off')
#            spl.imshow(img)
#
#            spl.plot(hbbox[0,:], hbbox[1,:], c='green')
#            spl.plot(obbox[0,:], obbox[1,:], c='blue')
#            spl.plot(conn[:,1], conn[:,0], c='red')
#            spl.scatter(conn[:,1], conn[:,0], c='red', s=5)
#            f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

def drawOverlapHoI(img, hbboxes, obboxes, props, imageMeta, imageDims, cfg, obj_mapping, hoi_mapping):
#    f, spl = plt.subplots(2,2)
#    spl = spl.ravel()
#    spl[0].imshow(img)
    import filters_helper as helper    
    gt_hbboxes, gt_obboxes = helper._transformGTBBox(imageMeta['objects'], obj_mapping, np.array(imageMeta['rels']), scale=imageDims['scale'], dosplit=True)
    gt_rels = helper._getRealRels(np.array(imageMeta['rels']))
    gt_objs = np.unique(gt_obboxes[:,4])
    
    inv_obj_mapping = {x:key for key,x in obj_mapping.items()}
    colours = ['#FEc75c', '#123456','#456789', '#abcdef','#fedcba', '#987654','#654321', '#994ee4']
    idxs = []
    nb_pairs = hbboxes.shape[0]
    c_idx=0

    for idx in range(nb_pairs):
#        if idx > 5 and idx < 100:
#            continue
#        if idx > 105:
#            break
#        hprop = (hbboxes[idx,4])
#        oprop = (obboxes[idx,4])
#        hlbl = int(hbboxes[idx,5])
#        olbl = int(obboxes[idx,5])
        hoilabels = np.where(props[idx,:]>0.01)[0]
#        hoilabels = list(range(600))
#        hoilabels = [x for x in hoilabels if obj_mapping[hoi_mapping[x]['obj']] in gt_objs]
        hoiprops = ([props[idx,x] for x in hoilabels if x in hoilabels])
        
        hbbox = hbboxes[idx,:4]*16
        obbox = obboxes[idx,:4]*16
        
        hbbox_cp = np.copy(hbbox)
        obbox_cp = np.copy(obbox)
        
        hbbox_cp[2] += hbbox_cp[0]
        hbbox_cp[3] += hbbox_cp[1]
        obbox_cp[2] += obbox_cp[0]
        obbox_cp[3] += obbox_cp[1]
        
        overlap = 0
        overlap_labels = []
        for gt_idx, gt_rel in enumerate(gt_rels):
            gt_hbbox = gt_hbboxes[gt_rel[0]:gt_rel[0]+1]
            gt_obbox = gt_obboxes[gt_rel[1]:gt_rel[1]+1]
            gt_label = gt_rel[2]
            
            if gt_label not in hoilabels:
#                print(gt_label, hoilabels)
                continue
                        
            hum_overlap = helper._computeIoUs(hbbox_cp, gt_hbbox)[0]
            obj_overlap = helper._computeIoUs(obbox_cp, gt_obbox)[0]
            
#            print(', '.join(['%.4f' % x for x in hbbox_cp]), ', '.join(['%.4f' % x for x in obbox_cp]))
#            print(', '.join(['%.4f' % x for x in gt_hbbox[0]]), ', '.join(['%.4f' % x for x in gt_obbox[0]]))
#            print(hum_overlap, obj_overlap)
            
            if min(hum_overlap, obj_overlap) > overlap:
                overlap = min(hum_overlap, obj_overlap)
            if min(hum_overlap, obj_overlap) > 0.5:
                overlap_labels.append(gt_label)
                
#        gt_hbboxes[:,2] -= gt_hbboxes[:,0]
#        gt_hbboxes[:,3] -= gt_hbboxes[:,1]
#        gt_obboxes[:,2] -= gt_obboxes[:,0]
#        gt_obboxes[:,3] -= gt_obboxes[:,1]
                
                
        hdotx = hbbox[0] + (hbbox[2] / 2)
        hdoty = hbbox[1] + (hbbox[3] / 2)
        odotx = obbox[0] + (obbox[2] / 2)
        odoty = obbox[1] + (obbox[3] / 2)
        conn = np.array([[odoty, odotx], [hdoty, hdotx]])
        hbbox = drawProposalBox(hbbox)
        obbox = drawProposalBox(obbox)        
        
        if len(hoilabels)>0 and overlap >= 0.5:
#            hoiprops = [props[idx,x] for x in hoilabels if x in overlap_labels]
            
            f, spl = plt.subplots(1,1)
            spl.axis('off')
            spl.imshow(img)
            c = colours[c_idx]
                        
            spl.plot(hbbox[0,:], hbbox[1,:], c='green')
            spl.plot(obbox[0,:], obbox[1,:], c='blue')
            spl.plot(conn[:,1], conn[:,0], c='red')
            spl.scatter(conn[:,1], conn[:,0], c='red', s=5)
            f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            
            
            pos_lbls = []
            neg_lbls = []
            for lbl in hoilabels:
                if lbl in overlap_labels:
                    pos_lbls.append(lbl)
                else:
                    neg_lbls.append(lbl)
            
#            hbbox = drawProposalBox(gt_hbbox[0])
#            obbox = drawProposalBox(gt_obbox[0])
#            spl[0].plot(hbbox[0,:], hbbox[1,:], c=c)
#            spl[0].plot(obbox[0,:], obbox[1,:], c=c)
            
            idxs.append(idx)
            labelsstr = ', '.join([str(x) for x in pos_lbls])
            print('Pos. label:', labelsstr, props[idx,pos_lbls])
            labelsstr = ', '.join([str(x) for x in neg_lbls])
            print('Neg. label:', labelsstr, props[idx,neg_lbls])
            
            c_idx = (c_idx+1) % len(colours)
        elif len(hoilabels) > 0:
            labelsstr = ', '.join([str(x) for x in hoilabels])
            print('All neg. label:', labelsstr, props[idx,hoilabels])
#            continue
            f, spl = plt.subplots(1,1)
            spl.axis('off')
            spl.imshow(img)
            spl.plot(hbbox[0,:], hbbox[1,:], c='green')
            spl.plot(obbox[0,:], obbox[1,:], c='blue')
            spl.plot(conn[:,1], conn[:,0], c='red')
            spl.scatter(conn[:,1], conn[:,0], c='red', s=5)
            f.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
#            hoiprops = ', '.join(['%.4f' % x for x in hoiprops])
#            print('Neg. label:', overlap, labelsstr, props[hoilabels])
    return np.array(idxs)

def drawPositiveCropHoI(hbboxes, obboxes, hcrops, ocrops, patterns, props, imageMeta, imageDims, cfg, obj_mapping):
    inv_obj_mapping = {x:key for key,x in obj_mapping.items()}
#    idxs = np.where(props[:,:]>0.5)[0]
    idxs = list(range(hbboxes.shape[0]))
    nb_pairs = len(idxs)
    
    f, spl = plt.subplots(4,8)
    spl = spl.ravel()
    
    hcrops += cfg.PIXEL_MEANS
    hcrops = hcrops.astype(np.uint8)
    ocrops += cfg.PIXEL_MEANS
    ocrops = ocrops.astype(np.uint8)
    
    for i, idx in enumerate(idxs):
        j = i*4
#        hprop = (hbboxes[idx,4]) if hbboxes is not None else -1
#        oprop = (obboxes[idx,4]) if obboxes is not None else -1
#        hlbl = int(hbboxes[idx,5]) if hbboxes is not None else 0
#        olbl = int(obboxes[idx,5]) if obboxes is not None else 0
#        hoiprop = np.where(props[idx,:]>0.5)[0]
        spl[j].imshow(hcrops[idx,::])
        spl[j+1].imshow(ocrops[idx,::])
        spl[j+2].imshow(patterns[idx,:,:,0])
        spl[j+3].imshow(patterns[idx,:,:,1])
#        print('Pos. label:', inv_obj_mapping[hlbl], inv_obj_mapping[olbl], hprop, oprop, hoiprop)
        if i == 7:
            break
            
    return np.array(idxs)   

def drawBoundingBox(bb):
    xmin = bb['xmin']; xmax = bb['xmax']
    ymin = bb['ymin']; ymax = bb['ymax']
    
    box = np.array([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]]).astype(int)
    return box

def drawProposalBox(bb):    
    xmin = bb[0]; xmax = bb[1]
    ymin = bb[2]; ymax = bb[3]
    
    xmin = bb[0]; ymin = bb[1]
    xmax = bb[2]; ymax = bb[3]

    xmin = bb[0]; xmax = xmin + bb[2]
    ymin = bb[1]; ymax = ymin + bb[3]
    
    box = np.array([[xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]]).astype(int)
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
    imageMeta = cp.copy(imageMeta)
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
        obj = all_objs[rel[1]]
        prs = all_objs[rel[0]]
        objBB = drawBoundingBox(obj)
        prsBB = drawBoundingBox(prs)
        
        prsC = [prs['ymin']+(prs['ymax']-prs['ymin'])/2, prs['xmin']+(prs['xmax']-prs['xmin'])/2]
        objC = [obj['ymin']+(obj['ymax']-obj['ymin'])/2, obj['xmin']+(obj['xmax']-obj['xmin'])/2]
        line = np.array([prsC, objC])
        objs.append(objBB)
        prss.append(prsBB)
        lines.append(line)
        names.append(label)
        
    nametitle = ', '.join([x['pred_ing'] for x in names]) + ' ' + names[0]['obj']
    f, spl = plt.subplots(2,2)
    spl = spl.ravel()
    for j, _ in enumerate(imageMeta['rels']):
        obj = objs[j]
        prs = prss[j]
        line = lines[j]
        spl[j].axis('off')
        spl[j].imshow(img)
        spl[j].plot(obj[0,:], obj[1,:], c='blue')
        spl[j].plot(prs[0,:], prs[1,:], c='green')
        spl[j].plot(line[:,1], line[:,0], c='red')
        spl[j].scatter(line[:,1], line[:,0], c='red', s=5)
#        print(names[j])
        if j == 3:
            break
    f.suptitle('GT: ' + nametitle)

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