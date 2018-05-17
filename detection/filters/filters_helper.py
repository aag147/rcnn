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
import cv2 as cv
import math

def unnormalizeRoIs(norm_rois, imageDims):
    shape = imageDims['output_shape']

    rois = np.zeros_like(norm_rois)
    rois[:,0] = norm_rois[:,0] * (shape[0]-0)
    rois[:,1] = norm_rois[:,1] * (shape[1]-0)
    rois[:,2] = norm_rois[:,2] * (shape[0]-0)
    rois[:,3] = norm_rois[:,3] * (shape[1]-0)

    return rois

def normalizeRoIs(rois, imageDims):
    shape = imageDims['output_shape']

    norm_rois = np.zeros_like(rois)
    norm_rois[:,0] = rois[:,0] / (shape[0]-0)
    norm_rois[:,1] = rois[:,1] / (shape[1]-0)
    norm_rois[:,2] = rois[:,2] / (shape[0]-0)
    norm_rois[:,3] = rois[:,3] / (shape[1]-0)

    return norm_rois


def deltas2Boxes(props, deltas, rois, cfg):
    bbox_threshold = cfg.hoi_bbox_threshold
    nb_rois_in_batch = props.shape[1]
    bboxes = {}
    boxes = []
    
    for ii in range(nb_rois_in_batch):
        if np.max(props[0, ii, :]) < bbox_threshold or np.argmax(props[0, ii, :]) == 0:
            continue
        
        labelID = np.argmax(props[0, ii, :])
        if labelID not in bboxes:
            bboxes[labelID] = []
        prop = np.max(props[0, ii, :])
        roi = rois[ii, :]
        
        regr = deltas[0, ii, 4 * labelID:4 * (labelID + 1)]
#            tx /= cfg.classifier_regr_std[0]
#            ty /= cfg.classifier_regr_std[1]
#            tw /= cfg.classifier_regr_std[2]
#            th /= cfg.classifier_regr_std[3]
        x, y, w, h = apply_regr(roi, regr)
#        print('rt',roi[0], roi[1], roi[2], roi[3])
#        print('dl',regr[0], regr[1], regr[2], regr[3])
#        print('bb',x, y, w, h)
#        bboxes[labelID].append([x, y, (x + w), (y + h), prop])
        boxes.append([x, y, w, h, prop, labelID])

    boxes = np.array(boxes)
    return boxes

def non_max_suppression_boxes(bboxes, cfg):
    # add some nms to reduce many boxes
    new_bboxes = []
    labelIDs = bboxes[:,5]
    for i in range(cfg.nb_object_classes):
        idxs = np.where(labelIDs == i)[0]
        sub_bboxes = bboxes[idxs,:]
        if sub_bboxes.shape[0] == 0:
            continue
        boxes_nms = non_max_suppression_fast(sub_bboxes, overlap_thresh=cfg.hoi_nms_overlap_thresh)
        new_bboxes.extend(boxes_nms)
        
    new_bboxes = np.array(new_bboxes)
    return new_bboxes

def deltas2Anchors(props, deltas, cfg, imageDims):
    # Deltas to coordinates by way of anchors
    # [dx,dy,dw,dh] -> (xmin,ymin,xmax,ymax)
    
    assert props.shape[0] == 1
    shape = imageDims['redux_shape']
    
    anc_idx = 0
    
    (rows, cols) = props.shape[1:3]
    A = np.zeros((4, props.shape[1], props.shape[2], props.shape[3]))

    for anchor_size in cfg.anchor_sizes:
        for anchor_ratio in cfg.anchor_ratios:
            
            anchor_x = (anchor_size * anchor_ratio[0]) #/ cfg.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) #/ cfg.rpn_stride
            
            regr = deltas[0, :, :, 4 * anc_idx:4 * anc_idx + 4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            A[0, :, :, anc_idx] = (X+0.5) * float(cfg.rpn_stride) - anchor_x / 2
            A[1, :, :, anc_idx] = (Y+0.5) * float(cfg.rpn_stride) - anchor_y / 2
            A[2, :, :, anc_idx] = anchor_x
            A[3, :, :, anc_idx] = anchor_y

            A[:, :, :, anc_idx] = apply_regr_np(A[:, :, :, anc_idx], regr)

            A[2, :, :, anc_idx] = np.maximum(1, A[2, :, :, anc_idx])
            A[3, :, :, anc_idx] = np.maximum(1, A[3, :, :, anc_idx])

            A[0, :, :, anc_idx] = np.maximum(0, A[0, :, :, anc_idx])
            A[1, :, :, anc_idx] = np.maximum(0, A[1, :, :, anc_idx])
            A[2, :, :, anc_idx] = np.minimum(shape[1] - 1 - A[0, :, :, anc_idx], A[2, :, :, anc_idx])
            A[3, :, :, anc_idx] = np.minimum(shape[0] - 1 - A[1, :, :, anc_idx], A[3, :, :, anc_idx])

            anc_idx += 1

    all_boxes = A.reshape((4, -1)).transpose((1, 0))    
    all_probs = props.reshape((-1))
    all_boxes /= float(cfg.rpn_stride)
    
    # Remove badly defined boxes
    xmin = all_boxes[:, 0]
    ymin = all_boxes[:, 1]
    w = all_boxes[:, 2]
    h = all_boxes[:, 3]

    ids = np.where((w <= 0) | (h <= 0))

    all_boxes = np.delete(all_boxes, ids, 0)
    all_probs = np.delete(all_probs, ids, 0)
    
    all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs])))
    
    return all_boxes


def get_GT_deltas(gt, at):
    # Centered coordinates
    cxg = (gt['xmin'] + gt['xmax']) / 2.0
    cyg = (gt['ymin'] + gt['ymax']) / 2.0
    cxa = (at['xmin'] + at['xmax']) / 2.0
    cya = (at['ymin'] + at['ymax']) / 2.0
    # Widths and heights
    wg  = gt['xmax'] - gt['xmin']
    hg  = gt['ymax'] - gt['ymin']
    wa  = at['xmax'] - at['xmin']
    ha  = at['ymax'] - at['ymin']
    #Target ground truth
    tx = (cxg - cxa) / float(wa)
    ty = (cyg - cya) / float(ha)
    tw = np.log(wg / float(wa))
    th = np.log(hg / float(ha))
    
    return tx, ty, tw, th
    
def apply_regr(roi, delta):
    try:
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]

        tx = delta[0]
        ty = delta[1]
        tw = delta[2]
        th = delta[3]
                
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
#        x1 = int(round(x1))
#        y1 = int(round(y1))
#        w1 = int(round(w1))
#        h1 = int(round(h1))

#        print(x1,y1,w1,h1)
        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h

def apply_regr_np(anchors, deltas):
    try:
        x = anchors[0, :, :]
        y = anchors[1, :, :]
        w = anchors[2, :, :]
        h = anchors[3, :, :]

        tx = deltas[0, :, :]
        ty = deltas[1, :, :]
        tw = deltas[2, :, :]
        th = deltas[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return anchors


def non_max_suppression_fast(boxes, overlap_thresh=0.5, max_boxes=300):
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    # TODO: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob] format!!!! with prob built in

    
    if boxes.shape[0] == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w  = boxes[:, 2]
    h  = boxes[:, 3]
    x2 = x1+w
    y2 = y1+h

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1) * (y2 - y1)
    # sorted by boxes last element which is prob
    indexes = np.argsort([i[-1] for i in boxes])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick]
    return boxes


def reduce_rois(true_labels, cfg):
    #############################
    ######## Parameters #########
    #############################    
    nb_detection_rois = cfg.nb_detection_rois
    
    neg_samples = np.where(true_labels[0, :, 0] == 1)
    pos_samples = np.where(true_labels[0, :, 0] == 0)

    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []

    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    # Half positives, half negatives
    if len(pos_samples) < nb_detection_rois // 2:
        selected_pos_samples = pos_samples.tolist()
    else:
        selected_pos_samples = np.random.choice(pos_samples, nb_detection_rois // 2, replace=False).tolist()
    try:
        selected_neg_samples = np.random.choice(neg_samples, nb_detection_rois - len(selected_pos_samples),
                                                replace=False).tolist()
    except:
        selected_neg_samples = np.random.choice(neg_samples, nb_detection_rois - len(selected_pos_samples),
                                                replace=True).tolist()

    sel_samples = selected_pos_samples + selected_neg_samples
    return sel_samples
       
def preprocessImage(img, cfg):
    shape = img.shape
    if shape[0] < shape[1]:
        newHeight = cfg.mindim
        scale = float(newHeight) / shape[0]
        newWidth = int(shape[1] * scale)
        if newWidth > cfg.maxdim:
            newWidth = cfg.maxdim
            scale = newWidth / shape[1]
            newHeight = shape[0] * scale
    else:
        newWidth = cfg.mindim
        scale = float(newWidth) / shape[1]
        newHeight = int(shape[0] * scale)
        if newHeight > cfg.maxdim:
            newHeight = cfg.maxdim
            scale = newHeight / shape[0]
            newWidth = shape[1] * scale


    newWidth = round(newWidth)
    newHeight = round(newHeight)

    scaleWidth = float(newWidth) / img.shape[1]
    scaleHeight = float(newHeight) / img.shape[0]
    scales = [scaleHeight, scaleWidth]
    # Rescale
    img = cv.resize(img, (newWidth, newHeight)).astype(np.float32)
    # Normalize
    img = (img - np.min(img)) / np.max(img)
    # Transpose
    img = img.transpose(cfg.order_of_dims)
    return img, scales

       
def normalizeGTboxes(gtboxes, scale=[1,1], rpn_stride=1, shape=[1,1], roundoff=False):
    gtnormboxes = []
    for relID, bbox in enumerate(gtboxes):
#        print(bbox)
        # get the GT box coordinates, and resize to account for image resizing
        xmin = ((bbox['xmin']-1) * scale[0] / rpn_stride) / shape[0]
        xmax = ((bbox['xmax']-1) * scale[0] / rpn_stride) / shape[0]
        ymin = ((bbox['ymin']-1) * scale[1] / rpn_stride) / shape[1]
        ymax = ((bbox['ymax']-1) * scale[1] / rpn_stride) / shape[1]
        if roundoff:
            xmin=int(round(xmin)); xmax=int(round(xmax))
            ymin=int(round(ymin)); ymax=int(round(ymax))
        gtnormboxes.append({'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax})    
    return gtnormboxes
