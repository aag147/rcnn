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

import utils
import filters_helper as helper

def prepareTargets(objpreds, bboxes, imageMeta, imageDims, cfg):    
    #############################
    ########## Image ############
    #############################
    hbboxes = imageMeta['humans']
    obboxes = imageMeta['objects']
    rels    = imageMeta['rels']
    
    scale = imageDims['scale']
    shape = imageDims['shape']
    
    #############################
    ###### Set Parameters #######
    #############################  
    nb_hoi_rois         = cfg.nb_hoi_rois
    nb_hoi_classes      = cfg.nb_hoi_classes
    hoi_max_overlap     = cfg.hoi_max_overlap
    hoi_min_overlap     = cfg.hoi_min_overlap
    nb_hoi_positives    = cfg.nb_hoi_positives
    nb_hoi_negatives1   = cfg.nb_hoi_negatives1
    nb_hoi_negatives2   = cfg.nb_hoi_negatives2
    rpn_stride          = cfg.rpn_stride
    
    #############################
    ########## bboxes ###########
    #############################
    hpredlabels = []
    hpredbboxes = []
    opredlabels = []
    opredbboxes = []
    
    for predidx in range(objpreds.shape[0]):
#        objpred = np.argmax(objpreds[predidx,:])
        objpred = objpreds[predidx,0]
        objbbox = bboxes[predidx, 4*objpred:4*objpred+4]
        if objpred == 1: #human
            hpredlabels.append(objpred)
            hpredbboxes.append(objbbox)
        elif objpred > 1: #object
            opredlabels.append(objpred)
            opredbboxes.append(objbbox)
            
    hpredlabels = np.array(hpredlabels)
    hpredbboxes = np.array(hpredbboxes)
    opredlabels = np.array(opredlabels)
    opredbboxes = np.array(opredbboxes)

    
    #############################
    #### Initialize matrices ####
    #############################
    combinations = np.array([len(hpredlabels), len(opredlabels)])
    combinations = {i:j for j in range(len(opredlabels)) for i in range(len(hpredlabels))}
    combinations = {i:{j:[] for j in range(len(opredlabels))} for i in range(len(hpredlabels))}
    all_human_boxes = []
    all_object_boxes = []
    all_labels = []
    all_type = []
    
    Xh = np.zeros([nb_hoi_rois, 5])
    Xo = np.zeros([nb_hoi_rois, 5])
    Yc = np.zeros([nb_hoi_rois, nb_hoi_classes])
    Xi = np.zeros([10, 4])
    
    
    #############################
    ##### Ground truth boxes ####
    #############################
    gthboxes = helper.normalizeGTboxes(hbboxes, scale=scale, rpn_stride=rpn_stride)
    gtoboxes = helper.normalizeGTboxes(obboxes, scale=scale, rpn_stride=rpn_stride)
    
    olabels = [obj['label'] for obj in obboxes]
    #############################
    ## Objects 2 Ground Truths ##
    #############################     
    human_overlaps = np.zeros([len(hpredlabels), len(gthboxes)])
    for hidx in range(hpredbboxes.shape[0]):
        (xmin, ymin, xmax, ymax) = hpredbboxes[hidx, :]
        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))
        ht = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        for gthidx, gth in enumerate(gthboxes):
            curr_iou = utils.get_iou(gth, ht)
            human_overlaps[hidx, gthidx] = curr_iou
        
    object_overlaps = np.zeros([len(opredlabels), len(gtoboxes)])
    for oidx in range(opredbboxes.shape[0]):
        (xmin, ymin, xmax, ymax) = opredbboxes[oidx, :]
        ot = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        print('pred', ot)
        for gtoidx, gto in enumerate(gtoboxes):
            print('truh', gto)
            curr_iou = utils.get_iou(gto, ot)
            if opredlabels[oidx]==olabels[gtoidx]:
                overlap = curr_iou
            elif opredlabels[oidx] in olabels:
                overlap = 0.0
            else:
                overlap = -2
            object_overlaps[oidx, gtoidx] = overlap
    
    print(human_overlaps)
    print(object_overlaps)
    #############################
    ##### Produce H/O pairs #####
    #############################         
    for relidx, rel in enumerate(rels):
        gthidx = rel[0]; gtoidx = rel[1]
        hoi_label = rel[2]
        for hidx, hiou in enumerate(human_overlaps[:,gthidx]):
            for oidx, oiou in enumerate(object_overlaps[:,gtoidx]):
                if hoi_max_overlap <= hiou and hoi_max_overlap <= oiou:
                    label = hoi_label
                elif hoi_min_overlap <= hiou and hoi_min_overlap <= oiou:
                    label = -1
                elif oiou==-2:
                    label = -2
                else:
                    label = -3
                if len(combinations[hidx][oidx])==1 and combinations[hidx][oidx][0]<0:
                    if label > combinations[hidx][oidx][0]:
                        combinations[hidx][oidx] = [label]
                    continue
                elif len(combinations[hidx][oidx])>0 and label < 0:
                    continue
                combinations[hidx][oidx].append(label)

                
    #############################
    ## Matrix pairs to vectors ##
    #############################    
    for hidx, oidxs in combinations.items():
        for oidx, labels in oidxs.items():
            if len(labels)==1 and labels[0] == -1:
                all_type.append(-1)
                labels = [0]
            elif len(labels)==1 and labels[0] == -2:
                all_type.append(-2)
                labels = [0]
            elif len(labels)==1 and labels[0] == -3:
                all_type.append(-3)
                labels = [0]
            elif len(labels)>0:
                if -1 in labels:
                    labels.remove(-1)
                all_type.append(1)
                labels = labels

            labels = utils.getMatrixLabels(nb_hoi_classes, [labels])
            labels = np.squeeze(labels)
            all_labels.append(labels)
            (xmin, ymin, xmax, ymax) = hpredbboxes[hidx, :]
            all_human_boxes.append([ymin/shape[0], xmin/shape[1], ymax/shape[0], xmax/shape[1]])
            (xmin, ymin, xmax, ymax) = opredbboxes[hidx, :]
            all_object_boxes.append([ymin/shape[0], xmin/shape[1], ymax/shape[0], xmax/shape[1]])
        
    all_human_boxes = np.array(all_human_boxes)
    all_object_boxes = np.array(all_object_boxes)
    all_labels = np.array(all_labels)
    all_type = np.array(all_type)

    ##############################
    # Three sample sources redux #
    ##############################
    valid_idxs = np.where(all_type != -3)
    positive_idxs = np.where(all_type==1)
    negative1_idxs = np.where(all_type==-1)
    negative2_idxs = np.where(all_type==-2)    
    
    return all_human_boxes[valid_idxs,:], all_object_boxes[valid_idxs,:], Xi, all_labels[valid_idxs,:], all_type
    
    if len(positive_idxs) > nb_hoi_positives:
        selected_pos_samples = np.random.choice(positive_idxs, nb_hoi_positives, replace=False)
    if len(negative1_idxs) > nb_hoi_negatives1:
        selected_neg1_samples = np.random.choice(negative1_idxs, nb_hoi_negatives1, replace=False)
    if len(negative2_idxs) > nb_hoi_negatives2:
        selected_neg2_samples = np.random.choice(negative2_idxs, nb_hoi_negatives2, replace=False)

    selected_samples = selected_pos_samples + selected_neg1_samples + selected_neg2_samples
    
    valid_human_boxes = valid_human_boxes[selected_samples,:]
    valid_object_boxes = valid_object_boxes[selected_samples,:]
    valid_labels = valid_labels[selected_samples,:]
    
    
    return valid_human_boxes, valid_object_boxes, Xi, valid_labels



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
    # add some nms to reduce many boxes
#    for cls_num, box in boxes.items():
#        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)
#        boxes[cls_num] = boxes_nms
#        print(class_mapping[cls_num] + ":")
#        for b in boxes_nms:
#            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
#            print('{} prob: {}'.format(b[0: 4], b[-1]))
