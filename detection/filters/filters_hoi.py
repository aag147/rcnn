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


def prepareTargets(bboxes, imageMeta, imageDims, cfg, class_mapping):    
    #############################
    ########## Image ############
    #############################
    gt_bboxes = imageMeta['objects']
    gt_rels    = imageMeta['rels']
    
    scale = imageDims['scale']
    shape = imageDims['shape']
    
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
    if hbboxes is None:
        return None, None, None, None
    
    
    #############################
    ##### Ground truth boxes ####
    #############################
    gthboxes, gtoboxes = helper._transformGTBBox(gt_bboxes, class_mapping, scale=scale, rpn_stride=cfg.rpn_stride)
    
    gt_relmap  = helper._getRelMap(gt_rels)
    
    #############################
    ## Unite humans and objects #
    #############################
    val_map = np.zeros((hbboxes.shape[0], obboxes.shape[0]))
    label_map = np.zeros((hbboxes.shape[0], obboxes.shape[0], cfg.nb_hoi_classes))
    hbb_map   = np.zeros((hbboxes.shape[0], obboxes.shape[0], 4))
    obb_map   = np.zeros((hbboxes.shape[0], obboxes.shape[0], 4))
    
    for hidx, hbox in enumerate(hbboxes):
        h_ious = helper._computeIoUs(hbox, gthboxes)
            
        for oidx, obox in enumerate(obboxes):
            objlabel = int(obox[4])
            o_ious = helper._computeIoUs(obox, gtoboxes)
                        
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
    final_vals = cp.copy(val_map[val_idxs[0], val_idxs[1]])
    final_labels = label_map[val_idxs[0], val_idxs[1]]
    final_hbbs   = hbb_map[val_idxs[0], val_idxs[1], :]
    final_obbs   = obb_map[val_idxs[0], val_idxs[1], :]
    
    
    
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
    
    
    if len(positive_idxs) > cfg.hoi_pos_share:
        selected_pos_samples = np.random.choice(positive_idxs, cfg.hoi_pos_share, replace=False)
        
    if len(negative1_idxs) > cfg.hoi_neg1_share:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.hoi_neg1_share, replace=False)
    elif len(negative1_idxs)>0:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.hoi_neg1_share, replace=True)
        
    if len(negative2_idxs) + len(selected_neg1_samples) + len(selected_pos_samples) > cfg.nb_hoi_rois:
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.hoi_neg2_share, replace=False)
    elif len(negative2_idxs) > 0:
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.nb_hoi_rois - len(selected_pos_samples) - len(selected_neg1_samples), replace=True)
    else:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.nb_hoi_rois - len(selected_pos_samples), replace=True)
        

    selected_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg2_samples.tolist()
    final_hbbs = final_hbbs[selected_samples,:]
    final_obbs = final_obbs[selected_samples,:]
    final_labels = final_labels[selected_samples,:]
    final_vals   = final_vals[selected_samples]
                
    
    return final_hbbs, final_obbs, final_labels, final_vals



########################
###### OUT DATED #######
########################

def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h

def deltas_to_bb(object_scores, object_deltas, rois, cfg):
    bbox_threshold = cfg.hoi_bbox_threshold
    nb_rois_in_batch = object_scores.shape[1]
    bboxes = []
    labels = []
    
    for ii in range(nb_rois_in_batch):
        if np.max(object_scores[0, ii, :]) < bbox_threshold or np.argmax(object_scores[0, ii, :]) == 0:
            continue

        labelID = np.argmax(object_scores[0, ii, :])
        prop = np.max(object_scores[0, ii, :])
        (batchID, x, y, w, h) = rois[0, ii, :]
        try:
            (tx, ty, tw, th) = object_deltas[0, ii, 4 * labelID:4 * (labelID + 1)]
#            tx /= cfg.classifier_regr_std[0]
#            ty /= cfg.classifier_regr_std[1]
#            tw /= cfg.classifier_regr_std[2]
#            th /= cfg.classifier_regr_std[3]
            x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
        except Exception as e:
            print(e)
            pass
        bboxes.append([x, y, (x + w), (y + h)])
        labels.append([labelID, prop])
        
    labels = np.array(labels)
    bboxes = np.array(bboxes)
    return labels, bboxes
    
    
    
    
######## CRAP #############
    #############################
    #### Initialize matrices ####
    #############################
#    combinations = np.array([len(hpredlabels), len(opredlabels)])
#    combinations = {i:j for j in range(len(opredlabels)) for i in range(len(hpredlabels))}
#    combinations = {i:{j:[] for j in range(len(opredlabels))} for i in range(len(hpredlabels))}
#    all_human_boxes = []
#    all_object_boxes = []
#    all_labels = []
#    all_type = []
#    
#    Xh = np.zeros([nb_hoi_rois, 5])
#    Xo = np.zeros([nb_hoi_rois, 5])
#    Yc = np.zeros([nb_hoi_rois, nb_hoi_classes])
#    Xi = np.zeros([10, 4])
#    
#    
#    olabels = [obj['label'] for obj in obboxes]
#    #############################
#    ## Objects 2 Ground Truths ##
#    #############################     
#    human_overlaps = np.zeros([len(hpredlabels), len(gthboxes)])
#    for hidx in range(hpredbboxes.shape[0]):
#        (xmin, ymin, xmax, ymax) = hpredbboxes[hidx, :]
#        xmin = int(round(xmin))
#        ymin = int(round(ymin))
#        xmax = int(round(xmax))
#        ymax = int(round(ymax))
#        ht = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
#        for gthidx, gth in enumerate(gthboxes):
#            curr_iou = utils.get_iou(gth, ht)
#            human_overlaps[hidx, gthidx] = curr_iou
#        
#    object_overlaps = np.zeros([len(opredlabels), len(gtoboxes)])
#    for oidx in range(opredbboxes.shape[0]):
#        (xmin, ymin, xmax, ymax) = opredbboxes[oidx, :]
#        ot = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
#        print('pred', ot)
#        for gtoidx, gto in enumerate(gtoboxes):
#            print('truh', gto)
#            curr_iou = utils.get_iou(gto, ot)
#            if opredlabels[oidx]==olabels[gtoidx]:
#                overlap = curr_iou
#            elif opredlabels[oidx] in olabels:
#                overlap = 0.0
#            else:
#                overlap = -2
#            object_overlaps[oidx, gtoidx] = overlap
#    
#    print(human_overlaps)
#    print(object_overlaps)
#    #############################
#    ##### Produce H/O pairs #####
#    #############################         
#    for relidx, rel in enumerate(rels):
#        gthidx = rel[0]; gtoidx = rel[1]
#        hoi_label = rel[2]
#        for hidx, hiou in enumerate(human_overlaps[:,gthidx]):
#            for oidx, oiou in enumerate(object_overlaps[:,gtoidx]):
#                if hoi_max_overlap <= hiou and hoi_max_overlap <= oiou:
#                    label = hoi_label
#                elif hoi_min_overlap <= hiou and hoi_min_overlap <= oiou:
#                    label = -1
#                elif oiou==-2:
#                    label = -2
#                else:
#                    label = -3
#                if len(combinations[hidx][oidx])==1 and combinations[hidx][oidx][0]<0:
#                    if label > combinations[hidx][oidx][0]:
#                        combinations[hidx][oidx] = [label]
#                    continue
#                elif len(combinations[hidx][oidx])>0 and label < 0:
#                    continue
#                combinations[hidx][oidx].append(label)
#
#                
#    #############################
#    ## Matrix pairs to vectors ##
#    #############################    
#    for hidx, oidxs in combinations.items():
#        for oidx, labels in oidxs.items():
#            if len(labels)==1 and labels[0] == -1:
#                all_type.append(-1)
#                labels = [0]
#            elif len(labels)==1 and labels[0] == -2:
#                all_type.append(-2)
#                labels = [0]
#            elif len(labels)==1 and labels[0] == -3:
#                all_type.append(-3)
#                labels = [0]
#            elif len(labels)>0:
#                if -1 in labels:
#                    labels.remove(-1)
#                all_type.append(1)
#                labels = labels
#
#            labels = utils.getMatrixLabels(nb_hoi_classes, [labels])
#            labels = np.squeeze(labels)
#            all_labels.append(labels)
#            (xmin, ymin, xmax, ymax) = hpredbboxes[hidx, :]
#            all_human_boxes.append([ymin/shape[0], xmin/shape[1], ymax/shape[0], xmax/shape[1]])
#            (xmin, ymin, xmax, ymax) = opredbboxes[hidx, :]
#            all_object_boxes.append([ymin/shape[0], xmin/shape[1], ymax/shape[0], xmax/shape[1]])
#        
#    all_human_boxes = np.array(all_human_boxes)
#    all_object_boxes = np.array(all_object_boxes)
#    all_labels = np.array(all_labels)
#    all_type = np.array(all_type)
#
#    ##############################
#    # Three sample sources redux #
#    ##############################
#    valid_idxs = np.where(all_type != -3)
#    positive_idxs = np.where(all_type==1)
#    negative1_idxs = np.where(all_type==-1)
#    negative2_idxs = np.where(all_type==-2)    
#    
#    return all_human_boxes[valid_idxs,:], all_object_boxes[valid_idxs,:], Xi, all_labels[valid_idxs,:], all_type
#    
#    if len(positive_idxs) > nb_hoi_positives:
#        selected_pos_samples = np.random.choice(positive_idxs, nb_hoi_positives, replace=False)
#    if len(negative1_idxs) > nb_hoi_negatives1:
#        selected_neg1_samples = np.random.choice(negative1_idxs, nb_hoi_negatives1, replace=False)
#    if len(negative2_idxs) > nb_hoi_negatives2:
#        selected_neg2_samples = np.random.choice(negative2_idxs, nb_hoi_negatives2, replace=False)
#
#    selected_samples = selected_pos_samples + selected_neg1_samples + selected_neg2_samples
#    
#    valid_human_boxes = all_human_boxes[selected_samples,:]
#    valid_object_boxes = all_object_boxes[selected_samples,:]
#    valid_labels = all_labels[selected_samples,:]
#    
#    
#    return valid_human_boxes, valid_object_boxes, Xi, valid_labels