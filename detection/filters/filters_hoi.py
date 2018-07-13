# -*- coding: utf-8 -*-
"""
Created on Mon May  7 08:55:19 2018

@author: aag14
"""

import sys 
sys.path.append('../../')
sys.path.append('../shared/')
sys.path.append('../models/')
sys.path.append('../cfgs/')
sys.path.append('../layers/')

import numpy as np
import random
import cv2 as cv
import math
import copy as cp

import utils
import filters_helper as helper

#######################
#### PROCESS INPUT ####
#######################
def unprepareInputs(h_bboxes_norm, o_bboxes_norm, imageDims):
    #(idx,ymin,xmin,ymax,xmax) -> (xmin,ymin,width,height)
    
    #Humans
    h_bboxes = cp.copy(h_bboxes_norm)
    h_bboxes = h_bboxes[:,:,1:]
    h_bboxes = h_bboxes[:,:,(1,0,3,2)]

    h_bboxes[:,:,2] = h_bboxes[:,:,2] - h_bboxes[:,:,0]
    h_bboxes[:,:,3] = h_bboxes[:,:,3] - h_bboxes[:,:,1]
    
    h_bboxes = helper.unnormalizeRoIs(h_bboxes, imageDims)
    
    #Objects
    o_bboxes = cp.copy(o_bboxes_norm)
    o_bboxes = o_bboxes[:,:,1:]
    o_bboxes = o_bboxes[:,:,(1,0,3,2)]

    o_bboxes[:,:,2] = o_bboxes[:,:,2] - o_bboxes[:,:,0]
    o_bboxes[:,:,3] = o_bboxes[:,:,3] - o_bboxes[:,:,1]
    
    o_bboxes = helper.unnormalizeRoIs(o_bboxes, imageDims)
    return h_bboxes, o_bboxes

def prepareInputs(h_bboxes, o_bboxes, imageDims):
    #(xmin,ymin,width,height) -> (idx,ymin,xmin,ymax,xmax)
    #Humans
    newh_bboxes = cp.copy(h_bboxes)
    
    newh_bboxes = newh_bboxes[:,:,(1,0,3,2)]
    newh_bboxes[:,:,2] = newh_bboxes[:,:,2] + newh_bboxes[:,:,0]
    newh_bboxes[:,:,3] = newh_bboxes[:,:,3] + newh_bboxes[:,:,1]
    
    newh_bboxes = helper.normalizeRoIs(newh_bboxes, imageDims)
    newh_bboxes = np.insert(newh_bboxes, 0, 0, axis=2)
    
    #Objects
    newo_bboxes = cp.copy(o_bboxes)
    newo_bboxes = newo_bboxes[:,:,(1,0,3,2)]
    newo_bboxes[:,:,2] = newo_bboxes[:,:,2] + newo_bboxes[:,:,0]
    newo_bboxes[:,:,3] = newo_bboxes[:,:,3] + newo_bboxes[:,:,1]
    
    newo_bboxes = helper.normalizeRoIs(newo_bboxes, imageDims)
    newo_bboxes = np.insert(newo_bboxes, 0, 0, axis=2)    
    
    return newh_bboxes, newo_bboxes

def splitInputs(bboxes, imageMeta, obj_mapping):
    h_idxs = np.where(bboxes[0,:,5]==1)[0]
    o_idxs = np.where(bboxes[0,:,5]>0)[0]
    
    hbboxes = bboxes[:,h_idxs,:4]
    obboxes = bboxes[:,o_idxs,:]
    obboxes = knownObjects(obboxes, imageMeta, obj_mapping)
    
    nb_hbboxes = hbboxes.shape[1]
    nb_obboxes = obboxes.shape[1]
    
    bboxes_pairs = np.zeros((2, nb_hbboxes*nb_obboxes, 4))
    for idx in range(nb_obboxes):
        s_idx = idx*nb_hbboxes
        f_idx = s_idx + nb_hbboxes
        obbox = obboxes[0,idx:idx+1,:]
        obbox_list = np.repeat(obbox, nb_hbboxes, axis=0)
        bboxes_pairs[0,s_idx:f_idx,:] = hbboxes[0,::]
        bboxes_pairs[1,s_idx:f_idx,:] = obbox_list
    
    return bboxes_pairs[0:1,::], bboxes_pairs[1:2,::]
        
def knownObjects(bboxes, imageMeta, obj_mapping):
    objs = imageMeta['objects']
    lbls = []
    for rel in imageMeta['rels']:
        obj = objs[rel[1]]
        label = obj_mapping[obj['label']]
        if label not in lbls:
            lbls.append(label)
    known_idxs = np.in1d(bboxes[0,:,5], lbls)
    bboxes = bboxes[:,known_idxs,:4]
    return bboxes

def convertBB2Crop(img, h_bboxes, o_bboxes, imageDims):
    #(xmin,ymin,width,height) -> (ymin:ymax,xmin:xmax,3)
    img_shape = img[0].shape
    h_bboxes = np.copy(h_bboxes)
    o_bboxes = np.copy(o_bboxes)
    h_bboxes, o_bboxes = prepareInputs(h_bboxes, o_bboxes, imageDims)
    h_bboxes = h_bboxes[0,:,1:]
    o_bboxes = o_bboxes[0,:,1:]
    
    
    h_bboxes[:,0] *= img_shape[0]
    h_bboxes[:,1] *= img_shape[1]
    h_bboxes[:,2] *= img_shape[0]
    h_bboxes[:,3] *= img_shape[1]
    
    o_bboxes[:,0] *= img_shape[0]
    o_bboxes[:,1] *= img_shape[1]
    o_bboxes[:,2] *= img_shape[0]
    o_bboxes[:,3] *= img_shape[1]
    
    h_bboxes = h_bboxes.astype(np.uint32)
    o_bboxes = o_bboxes.astype(np.uint32)
    h_bboxes[:,0] = np.minimum(img_shape[0]-2, h_bboxes[:,0])
    h_bboxes[:,1] = np.minimum(img_shape[1]-2, h_bboxes[:,1])
    o_bboxes[:,0] = np.minimum(img_shape[0]-2, o_bboxes[:,0])
    o_bboxes[:,1] = np.minimum(img_shape[1]-2, o_bboxes[:,1])
    
    
    nb_interactions = h_bboxes.shape[0]
    
    hcrops = np.zeros((nb_interactions, 227, 227, 3))
    ocrops = np.zeros((nb_interactions, 227, 227, 3))

    for idx in range(nb_interactions):
        h_bbox = h_bboxes[idx,:]
        hcrop = np.copy(img[0, h_bbox[0]:h_bbox[2], h_bbox[1]:h_bbox[3],:])
        hcrop = cv.resize(hcrop, (227, 227), interpolation=cv.INTER_LINEAR)
        hcrops[idx,::] = hcrop
        
        o_bbox = o_bboxes[idx,:]
        ocrop = np.copy(img[0, o_bbox[0]:o_bbox[2], o_bbox[1]:o_bbox[3],:])
        ocrop = cv.resize(ocrop, (227, 227), interpolation=cv.INTER_LINEAR)
        ocrops[idx,::] = ocrop
    
    return hcrops, ocrops




#######################
### PROCESS TARGETS ###
#######################
def loadData(imageInput, imageDims, cfg):
    if imageInput is None:
        return None
    all_hbboxes = np.array(imageInput['hbboxes']).astype(np.float64) / 1000.0
    all_obboxes = np.array(imageInput['o_bboxes']).astype(np.float64) / 1000.0
    all_target_labels = (imageInput['hoi_labels'])
    val_map = np.array(imageInput['val_map'])
        
    all_target_labels = utils.getMatrixLabels(cfg.nb_hoi_classes, all_target_labels)
    
#    if len(np.where(val_map==3)[0])==0:
#        return None, None, None, None
    
    all_hbboxes = np.expand_dims(all_hbboxes, axis=0)
    all_obboxes = np.expand_dims(all_obboxes, axis=0)
    all_target_labels = np.expand_dims(all_target_labels, axis=0)
    val_map = np.expand_dims(val_map, axis=0)
    
    return all_hbboxes, all_obboxes, all_target_labels, val_map



def convertData(Y, cfg, mode='train'):
    [all_hbboxes, all_obboxes, all_target_labels, all_val_map] = Y    
    if mode=='train':
        sel_samples = filterTargets(all_val_map, None, 75, None, nb_neg2=150)
    else:
        sel_samples = filterTargets(all_val_map, None, None, None, nb_neg2=None)
        if len(sel_samples)==0:
            return None
    
    all_hbboxes = np.copy(all_hbboxes[0,sel_samples,:])
    all_obboxes = np.copy(all_obboxes[0,sel_samples,:])
    all_target_labels = np.copy(all_target_labels[0,sel_samples,:])
    
    all_val_map = np.copy(all_val_map[0,sel_samples])
    
    all_target_labels = [np.where(x==1)[0].tolist() for x in all_target_labels.astype(np.uint8)]
    all_hbboxes = [[round(x*1000) for x in box] for box in all_hbboxes.tolist()]
    all_obboxes = [[round(x*1000) for x in box] for box in all_obboxes.tolist()]
    all_val_map = all_val_map.astype(np.uint8).tolist()
    
    hoiMeta = {'hbboxes':all_hbboxes, 'o_bboxes':all_obboxes, 'hoi_labels':all_target_labels, 'val_map':all_val_map}
    return hoiMeta

def convertResults(hbboxes, obboxes, predicted_labels, imageMeta, scale, rpn_stride):
    hbboxes = np.copy(hbboxes)
    obboxes = np.copy(obboxes)
    
    # humans
    hbboxes[:,0] = ((hbboxes[:,0]) * rpn_stride / scale[0])
    hbboxes[:,1] = ((hbboxes[:,1]) * rpn_stride / scale[1])
    hbboxes[:,2] = ((hbboxes[:,2]) * rpn_stride / scale[0])
    hbboxes[:,3] = ((hbboxes[:,3]) * rpn_stride / scale[1])
    
    #(..,width,height) ->  (..,xmax,ymax)
    hbboxes[:,2] = hbboxes[:,2] + hbboxes[:,0]
    hbboxes[:,3] = hbboxes[:,3] + hbboxes[:,1]
    
    # objects
    obboxes[:,0] = ((obboxes[:,0]) * rpn_stride / scale[0])
    obboxes[:,1] = ((obboxes[:,1]) * rpn_stride / scale[1])
    obboxes[:,2] = ((obboxes[:,2]) * rpn_stride / scale[0])
    obboxes[:,3] = ((obboxes[:,3]) * rpn_stride / scale[1])
    
    #(..,width,height) ->  (..,xmax,ymax)
    obboxes[:,2] = obboxes[:,2] + obboxes[:,0]
    obboxes[:,3] = obboxes[:,3] + obboxes[:,1]
    
    results = []
    nb_boxes = hbboxes.shape[0]
    for bidx in range(nb_boxes):
        hbbox = hbboxes[bidx,:]
        obbox = obboxes[bidx,:]
        preds = predicted_labels[bidx]
        labels = np.where(preds>0.5)[0].tolist()
        props = preds[labels]
        nb_preds = len(labels)

        hbbox = [round(float(x),2) for x in hbbox.tolist()]
        obbox = [round(float(x),2) for x in obbox.tolist()]
        
        for pidx in range(nb_preds):
            label = labels[pidx]    
            prop = props[pidx]
            res = {'image_id': (imageMeta['id']), 'category_id': int(label), 'hbbox': hbbox, 'obbox': obbox, 'score': round(float(prop),4)}
            results.append(res)
    return results


def filterTargets(val_map, nb_pos, nb_neg1, nb_hoi_rois, nb_neg2=0):
    val_map = np.copy(val_map[0])
    positive_idxs = np.where(val_map==3)[0]
    negative1_idxs = np.where(val_map==2)[0]
    negative2_idxs = np.where(val_map==0)[0]
    negative3_idxs = np.where(val_map==1)[0]
    
    selected_pos_samples = positive_idxs
    selected_neg1_samples = negative1_idxs
    selected_neg2_samples = negative2_idxs
    selected_neg3_samples = negative3_idxs
    
    if nb_pos is None and nb_neg1 is None and nb_hoi_rois is None:
        sel_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg3_samples.tolist()    
        return sel_samples
    
    if nb_pos is not None and len(positive_idxs) > nb_pos and len(negative2_idxs)>0:
        selected_pos_samples = np.random.choice(positive_idxs, nb_pos, replace=False)
        
    if len(negative1_idxs) > nb_neg1 and len(negative2_idxs)>0:
        selected_neg1_samples = np.random.choice(negative1_idxs, nb_neg1, replace=False)
        
    if nb_neg2 > 0:
        if len(negative2_idxs) > nb_neg2:
            selected_neg2_samples = np.random.choice(negative2_idxs, nb_neg2, replace=False)
    elif len(negative2_idxs) + len(selected_neg1_samples) + len(selected_pos_samples) > nb_hoi_rois:
        selected_neg2_samples = np.random.choice(negative2_idxs, nb_hoi_rois - len(selected_pos_samples) - len(selected_neg1_samples), replace=False)
    elif len(negative2_idxs) > 0:
        selected_neg2_samples = np.random.choice(negative2_idxs, nb_hoi_rois - len(selected_pos_samples) - len(selected_neg1_samples), replace=True)
        
    sel_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg2_samples.tolist()    

    return sel_samples

def reduceTargets(Y, cfg, batchidx=None):
    #out: hbboxes [{1}, {batch_size}, (0,ymin,xmin,ymax,xmax)]
    #out: obboxes [{1}, {batch_size}, (0,ymin,xmin,ymax,xmax)]
    #out: labels [{1}, {batch_size}, {nb_hoi_classes}]
    [all_hbboxes, all_obboxes, all_target_labels, all_val_map] = Y
    
    hbboxes = np.zeros((1, cfg.nb_hoi_rois, 6))
    obboxes = np.zeros((1, cfg.nb_hoi_rois, 6))
    target_labels  = np.zeros((1, cfg.nb_hoi_rois, cfg.nb_hoi_classes))
    
    good_idxs = np.where((all_hbboxes[0,:,2]>0) & (all_hbboxes[0,:,3]>0) & (all_obboxes[0,:,2]>0) & (all_obboxes[0,:,3]>0))[0]
    
    all_hbboxes = all_hbboxes[:,good_idxs,:]
    all_obboxes = all_obboxes[:,good_idxs,:]
    all_target_labels = all_target_labels[:,good_idxs,:]
    all_val_map = all_val_map[:,good_idxs]
    
    
    ## Pick reduced indexes ##
    if batchidx is None:        
        sel_samples = filterTargets(all_val_map, cfg.hoi_pos_share, cfg.hoi_neg1_share, cfg.nb_hoi_rois)
        assert(len(sel_samples) == cfg.nb_hoi_rois)

    else:        
        sidx = batchidx * cfg.nb_hoi_rois
        fidx = min(all_target_labels.shape[1], sidx + cfg.nb_hoi_rois)
        sel_samples = list(range(sidx,fidx))
        
    assert(target_labels.shape[1] == cfg.nb_hoi_rois)
    
    ## Reduce data by picked indexes ##  
    hbboxes[:,:len(sel_samples),:]          = all_hbboxes[:, sel_samples, :]
    obboxes[:,:len(sel_samples),:]          = all_obboxes[:, sel_samples, :]
    target_labels[:,:len(sel_samples),:]    = all_target_labels[:, sel_samples, :]
    
    return hbboxes, obboxes, target_labels, all_val_map[:,sel_samples]

def createInteractionPatterns(hbbs, obbs, cfg):
    hbbs = np.copy(hbbs[0,::])
    obbs = np.copy(obbs[0,::])
    patterns = getDataPairWiseStream(hbbs, obbs, cfg)
    patterns = np.expand_dims(patterns, axis=0)
    return patterns

def createTargets(bboxes, imageMeta, imageDims, cfg, class_mapping):    
    #############################
    ########## Image ############
    #############################
    gt_bboxes = imageMeta['objects']
    gt_rels    = np.array(imageMeta['rels'])
    
    scale = imageDims['scale']
    shape = imageDims['shape']
    output_shape = imageDims['output_shape']
    
    #############################
    ###### Set Parameters #######
    #############################  
#    nb_hoi_rois         = cfg.nb_hoi_rois
#    nb_hoi_classes      = cfg.nb_hoi_classes
#    hoi_max_overlap     = cfg.hoi_max_overlap
#    hoi_min_overlap     = cfg.hoi_min_overlap
#    nb_hoi_positives    = cfg.nb_hoi_positives
#    nb_hoi_negatives1   = cfg.nb_hoi_negatives1
#    nb_hoi_negatives2   = cfg.nb_hoi_negatives2
#    rpn_stride          = cfg.rpn_stride
    
    #############################
    ########## bboxes ###########
    #############################    
    hbboxes, obboxes = helper._transformBBoxes(bboxes)
    if hbboxes is None or obboxes is None:
        return None, None, None, None
    
    
    #############################
    ##### Ground truth boxes ####
    #############################
    gthboxes, gtoboxes = helper._transformGTBBox(gt_bboxes, class_mapping, gt_rels, scale=scale, rpn_stride=cfg.rpn_stride)
    
    gt_relmap  = helper._getRelMap(gt_rels)
    
    #############################
    ## Unite humans and objects #
    #############################
    val_map = np.ones((hbboxes.shape[0], obboxes.shape[0])) * -1
    label_map = np.zeros((hbboxes.shape[0], obboxes.shape[0], cfg.nb_hoi_classes))
    hbb_map   = np.zeros((hbboxes.shape[0], obboxes.shape[0], 6))
    obb_map   = np.zeros((hbboxes.shape[0], obboxes.shape[0], 6))
    
#    print(gt_bboxes)
#    print(gthboxes)
#    print(gtoboxes)
    
    
    for hidx, hbox in enumerate(hbboxes):
        h_ious = helper._computeIoUs(hbox, gthboxes)
            
        for oidx, obox in enumerate(obboxes):
            objlabel = int(obox[5])
            o_ious = helper._computeIoUs(obox, gtoboxes)
                        
        
#            print(gt_relmap.shape, len(h_ious), len(o_ious))
            for gth_idx, h_iou in enumerate(h_ious):
                for gto_idx, o_iou in enumerate(o_ious):
                    gt_obj   = int(gtoboxes[gto_idx, 4])
                    gt_label = int(gt_relmap[gth_idx, gto_idx])
                    
                    if objlabel == gt_obj and h_iou >= cfg.hoi_max_overlap and o_iou >= cfg.hoi_max_overlap:
                        if gt_label >= 0:
                            # positive
#                            print('pos', objlabel, gt_label, h_iou, o_iou)
                            val_map[hidx, oidx] = 3
                            label_map[hidx, oidx, gt_label] = 1
                            hbb_map[hidx, oidx, :] = hbox[:6]
                            obb_map[hidx, oidx, :] = obox[:6]

                    elif objlabel == gt_obj and h_iou >= cfg.hoi_min_overlap and o_iou >= cfg.hoi_min_overlap and (h_iou < cfg.hoi_max_overlap or o_iou < cfg.hoi_max_overlap):
                        if val_map[hidx, oidx] < 2:
#                            print('neg1', objlabel, gt_label, h_iou, o_iou)
                            # negative1
                            val_map[hidx, oidx] = 2
                            hbb_map[hidx, oidx, :] = hbox[:6]
                            obb_map[hidx, oidx, :] = obox[:6]

                    
                            
                    elif objlabel == gt_obj:
                        if val_map[hidx, oidx] < 1:
                            # negative2
                            val_map[hidx, oidx] = 1
                            hbb_map[hidx, oidx, :] = hbox[:6]
                            obb_map[hidx, oidx, :] = obox[:6]   
                            
                    elif objlabel != gt_obj and o_iou < cfg.hoi_min_overlap:
                        if val_map[hidx, oidx] < 0:
                            # negative2
                            val_map[hidx, oidx] = 0
                            hbb_map[hidx, oidx, :] = hbox[:6]
                            obb_map[hidx, oidx, :] = obox[:6]
                     
                
                
    
    ##############################
    ### Reshape and remove bads ##
    ##############################
    val_idxs = np.where(val_map>=0)
#    print(val_idxs)
#    print(hbb_map.shape, obb_map.shape)
    final_vals = cp.copy(val_map[val_idxs[0], val_idxs[1]])
    final_labels = label_map[val_idxs[0], val_idxs[1]]
    final_hbbs   = hbb_map[val_idxs[0], val_idxs[1], :]
    final_obbs   = obb_map[val_idxs[0], val_idxs[1], :]
    
#    return val_map, hbb_map, obb_map, final_hbbs, final_obbs    
    
    # (_,_,xmax,ymax) -> (_,_,width,height)
    final_hbbs[:,2] -= final_hbbs[:,0]
    final_hbbs[:,3] -= final_hbbs[:,1]
    final_obbs[:,2] -= final_obbs[:,0]
    final_obbs[:,3] -= final_obbs[:,1]

    return np.expand_dims(final_hbbs, axis=0), np.expand_dims(final_obbs, axis=0), np.expand_dims(final_labels, axis=0), np.expand_dims(final_vals, axis=0)



#############################
#### Interaction Pattern ####
#############################
def _getSinglePairWiseStream(thisBB, thatBB, width, height, newWidth, newHeight, cfg):
    xmin = max(0, thisBB[0] - thatBB[0])
    xmax = width - max(0, thatBB[0]+thatBB[2] - (thisBB[0]+thisBB[2]))
    ymin = max(0, thisBB[1] - thatBB[1])
    ymax = height - max(0, thatBB[1]+thatBB[3] - (thisBB[1]+thisBB[3]))
        
    attWin = np.zeros([height,width])
    attWin[ymin:ymax, xmin:xmax] = 1

    
    attWin = cv.resize(attWin, (newWidth, newHeight), interpolation = cv.INTER_NEAREST)
    attWin = attWin.astype(np.int)

    xPad = int(abs(newWidth - cfg.winShape[1]) / 2)
    yPad = int(abs(newHeight - cfg.winShape[0]) / 2)
    attWinPad = np.zeros(cfg.winShape).astype(np.int)
#        print(attWin.shape, attWinPad.shape, xPad, yPad)
#        print(height, width, newHeight, newWidth)
    attWinPad[yPad:yPad+newHeight, xPad:xPad+newWidth] = attWin
    return attWinPad

def _getPairWiseStream(hbbox, obbox, cfg):
    width = max(hbbox[0]+hbbox[2], obbox[0]+obbox[2]) - min(hbbox[0], obbox[0])
    height = max(hbbox[1]+hbbox[3], obbox[1]+obbox[3]) - min(hbbox[1], obbox[1])
    if width > height:
        newWidth = cfg.winShape[1]
        apr = newWidth / width
        newHeight = max(1,int(height*apr))
    else:
        newHeight = cfg.winShape[0]
        apr = newHeight / height
        newWidth = max(1,int(width*apr))
        
    prsWin = _getSinglePairWiseStream(hbbox, obbox, width, height, newWidth, newHeight, cfg)
    objWin = _getSinglePairWiseStream(obbox, hbbox, width, height, newWidth, newHeight, cfg)
    
    return [prsWin, objWin]

def getDataPairWiseStream(hbboxes, obboxes, cfg):
    hbboxes *= cfg.rpn_stride
    hbboxes = hbboxes.astype(int)
    obboxes *= cfg.rpn_stride
    obboxes = obboxes.astype(int)
    
    dataPar = []
    for pairidx in range(hbboxes.shape[0]):
        relWin = _getPairWiseStream(hbboxes[pairidx], obboxes[pairidx], cfg)
        dataPar.append(relWin)
    dataPar = np.array(dataPar)
    dataPar = dataPar.transpose(cfg.par_order_of_dims)
    return dataPar