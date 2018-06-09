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


#######################
### PROCESS TARGETS ###
#######################
def loadData(imageInput, imageDims, cfg):
    if imageInput is None:
        return None, None, None, None
    all_hbboxes = np.array(imageInput['h_bboxes'])
    all_obboxes = np.array(imageInput['o_bboxes'])
    all_target_labels = (imageInput['hoi_labels'])
    val_map = np.array(imageInput['val_map'])
        
    all_target_labels = utils.getMatrixLabels(cfg.nb_hoi_classes, all_target_labels)
    
    if len(np.where(val_map==3)[0])==0:
        return None, None, None, None
    
    all_hbboxes = np.expand_dims(all_hbboxes, axis=0)
    all_obboxes = np.expand_dims(all_obboxes, axis=0)
    all_target_labels = np.expand_dims(all_target_labels, axis=0)
    val_map = np.expand_dims(val_map, axis=0)
    
    return all_hbboxes, all_obboxes, all_target_labels, val_map



def convertData(Y, cfg):
    [all_hbboxes, all_obboxes, all_target_labels, all_val_map] = Y
    all_target_labels = [np.where(x==1)[0].tolist() for x in all_target_labels.astype(int)]
    all_hbboxes = [[round(float(x), 2) for x in box] for box in all_hbboxes.tolist()]
    all_obboxes = [[round(float(x), 2) for x in box] for box in all_obboxes.tolist()]
    all_val_map = all_val_map.astype(int).tolist()
    
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


def reduceTargets(Y, cfg, batchidx=None):
    #out: hbboxes [{1}, {batch_size}, (0,ymin,xmin,ymax,xmax)]
    #out: obboxes [{1}, {batch_size}, (0,ymin,xmin,ymax,xmax)]
    #out: labels [{1}, {batch_size}, {nb_hoi_classes}]
    [all_hbboxes, all_obboxes, all_target_labels, val_map] = Y
    
    hbboxes = np.zeros((1, cfg.nb_hoi_rois, 4))
    obboxes = np.zeros((1, cfg.nb_hoi_rois, 4))
    target_labels  = np.zeros((1, cfg.nb_hoi_rois, cfg.nb_hoi_classes))
    
    ## Pick reduced indexes ##
    if batchidx is None:        
        positive_idxs = np.where(val_map[0]==3)[0]
        negative1_idxs = np.where(val_map[0]==2)[0]
        negative2_idxs = np.where(val_map[0]==1)[0]      
        
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
            
        sel_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg2_samples.tolist()
    else:        
        sidx = batchidx * cfg.nb_hoi_rois
        fidx = min(all_target_labels.shape[1], sidx + cfg.nb_hoi_rois)
        sel_samples = list(range(sidx,fidx))
        
    assert(target_labels.shape[1] == cfg.nb_hoi_rois)
    
    ## Reduce data by picked indexes ##  
    hbboxes[:,:len(sel_samples),:]          = all_hbboxes[:, sel_samples, :]
    obboxes[:,:len(sel_samples),:]          = all_obboxes[:, sel_samples, :]
    target_labels[:,:len(sel_samples),:]    = all_target_labels[:, sel_samples, :]
    
    return hbboxes, obboxes, target_labels

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