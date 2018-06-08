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

def loadData(imageInput, imageDims, cfg, batchidx = None):
    if imageInput is None:
        return None, None, None
    allh_bboxes = np.array(imageInput['h_bboxes'])
    allo_bboxes = np.array(imageInput['o_bboxes'])
    all_labels = (imageInput['hoi_labels'])
    val_map = np.array(imageInput['val_map'])
    all_labels = utils.getMatrixLabels(cfg.nb_hoi_classes, all_labels)
    
    if len(np.where(val_map==3)[0])==0:
        return None, None, None, None
    
    if batchidx is None:
        samples = reduce_hoi_rois(val_map, cfg)
        h_bboxes = allh_bboxes[samples, :]
        o_bboxes = allo_bboxes[samples, :]
        labels = all_labels[samples,:]
    else:
        h_bboxes = np.zeros((cfg.nb_hoi_rois, 4))
        o_bboxes = np.zeros((cfg.nb_hoi_rois, 4))
        labels  = np.zeros((cfg.nb_hoi_rois, cfg.nb_hoi_classes))
        patterns= np.zeros((cfg.nb_hoi_rois, cfg.winShape[0], cfg.winShape[1], 2))
        
        sidx = batchidx * cfg.nb_hoi_rois
        fidx = min(allh_bboxes.shape[1], sidx + cfg.nb_hoi_rois)
        h_bboxes[:,:fidx-sidx,:] = allh_bboxes[sidx:fidx, :]
        o_bboxes[:,:fidx-sidx,:] = allo_bboxes[sidx:fidx, :]
        labels[:,:fidx-sidx,:] = all_labels[sidx:fidx, :]
    
    labels = np.expand_dims(labels, axis=0)
    patterns = prepareInteractionPatterns(cp.copy(h_bboxes), cp.copy(o_bboxes), cfg)
    h_bboxes, o_bboxes = prepareInputs(h_bboxes, o_bboxes, imageDims)
    return h_bboxes, o_bboxes, patterns, labels

def reduce_hoi_rois(val_map, cfg):
    positive_idxs = np.where(val_map==3)[0]
    negative1_idxs = np.where(val_map==2)[0]
    negative2_idxs = np.where(val_map==1)[0]      
    
    selected_pos_samples = positive_idxs
    selected_neg1_samples = negative1_idxs
    selected_neg2_samples = negative2_idxs
    
    if len(positive_idxs) > cfg.hoi_pos_share:
        selected_pos_samples = np.random.choice(positive_idxs, cfg.hoi_pos_share, replace=False)
        
    if len(negative1_idxs) > cfg.hoi_neg1_share:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.hoi_neg1_share, replace=False)
    elif len(negative1_idxs)>0:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.hoi_neg1_share, replace=True)
        
    if len(negative2_idxs) + len(selected_neg1_samples) + len(selected_pos_samples) > cfg.nb_hoi_rois:
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.nb_hoi_rois - len(selected_pos_samples) - len(selected_neg1_samples), replace=False)
    elif len(negative2_idxs) > 0:
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.nb_hoi_rois - len(selected_pos_samples) - len(selected_neg1_samples), replace=True)
    else:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.nb_hoi_rois - len(selected_pos_samples), replace=True)
        
    selected_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg2_samples.tolist()
    
    
    assert(len(selected_samples) == 32)

    return selected_samples


def unprepareInputs(h_bboxes_norm, o_bboxes_norm, imageDims):
    #(idx,ymin,xmin,ymax,xmax) -> (xmin,ymin,width,height)
    
    #Humans
    h_bboxes = cp.copy(h_bboxes_norm)
    h_bboxes = h_bboxes[0,:,1:]
    h_bboxes = h_bboxes[:,(1,0,3,2)]

    h_bboxes[:,2] = h_bboxes[:,2] - h_bboxes[:,0]
    h_bboxes[:,3] = h_bboxes[:,3] - h_bboxes[:,1]
    
    h_bboxes = helper.unnormalizeRoIs(h_bboxes, imageDims)
    
    #Objects
    o_bboxes = cp.copy(o_bboxes_norm)
    o_bboxes = o_bboxes[0,:,1:]
    o_bboxes = o_bboxes[:,(1,0,3,2)]

    o_bboxes[:,2] = o_bboxes[:,2] - o_bboxes[:,0]
    o_bboxes[:,3] = o_bboxes[:,3] - o_bboxes[:,1]
    
    o_bboxes = helper.unnormalizeRoIs(o_bboxes, imageDims)
    return h_bboxes, o_bboxes

def prepareInputs(h_bboxes, o_bboxes, imageDims):
    #(xmin,ymin,width,height) -> (idx,ymin,xmin,ymax,xmax)
    #Humans
    newh_bboxes = cp.copy(h_bboxes)
    
    newh_bboxes = newh_bboxes[:,(1,0,3,2)]
    newh_bboxes[:,2] = newh_bboxes[:,2] + newh_bboxes[:,0]
    newh_bboxes[:,3] = newh_bboxes[:,3] + newh_bboxes[:,1]
    
    newh_bboxes = helper.normalizeRoIs(newh_bboxes, imageDims)

    newh_bboxes = np.insert(newh_bboxes, 0, 0, axis=1)
    newh_bboxes = np.expand_dims(newh_bboxes, axis=0)
    
    #Objects
    newo_bboxes = cp.copy(o_bboxes)
    newo_bboxes = newo_bboxes[:,(1,0,3,2)]
    newo_bboxes[:,2] = newo_bboxes[:,2] + newo_bboxes[:,0]
    newo_bboxes[:,3] = newo_bboxes[:,3] + newo_bboxes[:,1]
    
    newo_bboxes = helper.normalizeRoIs(newo_bboxes, imageDims)

    newo_bboxes = np.insert(newo_bboxes, 0, 0, axis=1)
    newo_bboxes = np.expand_dims(newo_bboxes, axis=0)
    
    return newh_bboxes, newo_bboxes

def prepareInteractionPatterns(final_hbbs, final_obbs, cfg):
    patterns = getDataPairWiseStream(final_hbbs, final_obbs, cfg)
    patterns = np.expand_dims(patterns, axis=0)
    return patterns

def prepareTargets(bboxes, imageMeta, imageDims, cfg, class_mapping):    
    #############################
    ########## Image ############
    #############################
    gt_bboxes = imageMeta['objects']
    gt_rels    = imageMeta['rels']
    
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
    val_map = np.zeros((hbboxes.shape[0], obboxes.shape[0]))
    label_map = np.zeros((hbboxes.shape[0], obboxes.shape[0], cfg.nb_hoi_classes))
    hbb_map   = np.zeros((hbboxes.shape[0], obboxes.shape[0], 4))
    obb_map   = np.zeros((hbboxes.shape[0], obboxes.shape[0], 4))
    
#    print(gt_bboxes)
#    print(gthboxes)
#    print(gtoboxes)
    
    for hidx, hbox in enumerate(hbboxes):
        h_ious = helper._computeIoUs(hbox, gthboxes)
            
        for oidx, obox in enumerate(obboxes):
            objlabel = int(obox[4])
            o_ious = helper._computeIoUs(obox, gtoboxes)
                        
        
#            print(gt_relmap.shape, len(h_ious), len(o_ious))
            for gth_idx, h_iou in enumerate(h_ious):
                for gto_idx, o_iou in enumerate(o_ious):
                    gt_obj   = int(gtoboxes[gto_idx, 4])
                    
                    if objlabel != gt_obj:
                        if val_map[hidx, oidx] < 1:
                            # negative2
                            val_map[hidx, oidx] = 1
                            hbb_map[hidx, oidx, :] = hbox[:4]
                            obb_map[hidx, oidx, :] = obox[:4]
                                                                                    
                    elif h_iou >= cfg.hoi_max_overlap and o_iou >= cfg.hoi_max_overlap:
                        
                        gt_label = int(gt_relmap[gth_idx, gto_idx])

                        if gt_label >= 0:
                            val_map[hidx, oidx] = 3
                            label_map[hidx, oidx, gt_label] = 1
                            hbb_map[hidx, oidx, :] = hbox[:4]
                            obb_map[hidx, oidx, :] = obox[:4]
                    elif h_iou >= cfg.hoi_min_overlap and o_iou >= cfg.hoi_min_overlap:
                        if val_map[hidx, oidx] < 2:
                            # negative1
                            val_map[hidx, oidx] = 2
                            hbb_map[hidx, oidx, :] = hbox[:4]
                            obb_map[hidx, oidx, :] = obox[:4]
                
                
    
    ##############################
    ### Reshape and remove bads ##
    ##############################
    val_idxs = np.where(val_map>0)
#    print(val_idxs)
#    print(hbb_map.shape, obb_map.shape)
    final_vals = cp.copy(val_map[val_idxs[0], val_idxs[1]])
    final_labels = label_map[val_idxs[0], val_idxs[1]]
    final_hbbs   = hbb_map[val_idxs[0], val_idxs[1], :]
    final_obbs   = obb_map[val_idxs[0], val_idxs[1], :]
    
#    return val_map, hbb_map, obb_map, final_hbbs, final_obbs
    
    # Crop to image boundary
    final_hbbs[:,0] = np.maximum(0.0, final_hbbs[:,0])
    final_hbbs[:,1] = np.maximum(0.0, final_hbbs[:,1])
    final_hbbs[:,2] = np.minimum(output_shape[1], final_hbbs[:,2])
    final_hbbs[:,3] = np.minimum(output_shape[0], final_hbbs[:,3])
    
    final_obbs[:,0] = np.maximum(0.0, final_obbs[:,0])
    final_obbs[:,1] = np.maximum(0.0, final_obbs[:,1])
    final_obbs[:,2] = np.minimum(output_shape[1], final_obbs[:,2])
    final_obbs[:,3] = np.minimum(output_shape[0], final_obbs[:,3])
    
    
    # (_,_,xmax,ymax) -> (_,_,width,height)
    final_hbbs[:,2] -= final_hbbs[:,0]
    final_hbbs[:,3] -= final_hbbs[:,1]
    final_obbs[:,2] -= final_obbs[:,0]
    final_obbs[:,3] -= final_obbs[:,1]
    
    
    ##############################
    # Three sample sources redux #
    ##############################
    positive_idxs = np.where(final_vals==3)[0]
    negative1_idxs = np.where(final_vals==2)[0]
    negative2_idxs = np.where(final_vals==1)[0]      
    
    selected_pos_samples = positive_idxs
    selected_neg1_samples = negative1_idxs
    selected_neg2_samples = negative2_idxs
    
    
    if len(negative1_idxs) > 24:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.hoi_neg1_share, replace=False)
        
    if len(negative2_idxs) > 32:
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.hoi_neg2_share, replace=False) 
    
    
    selected_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg2_samples.tolist()
    final_hbbs = final_hbbs[selected_samples,:]
    final_obbs = final_obbs[selected_samples,:]
    final_labels = final_labels[selected_samples,:]
    final_vals   = final_vals[selected_samples]
    
    return final_hbbs, final_obbs, final_labels, final_vals



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
        newHeight = int(height*apr) 
    else:
        newHeight = cfg.winShape[0]
        apr = newHeight / height
        newWidth = int(width*apr)
        
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