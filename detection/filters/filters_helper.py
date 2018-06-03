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
import copy as cp

def unnormalizeRoIs(norm_rois, imageDims):
    shape = imageDims['output_shape']

    rois = np.zeros_like(norm_rois)
    rois[:,0] = norm_rois[:,0] * (shape[1]-0)
    rois[:,1] = norm_rois[:,1] * (shape[0]-0)
    rois[:,2] = norm_rois[:,2] * (shape[1]-0)
    rois[:,3] = norm_rois[:,3] * (shape[0]-0)

    return rois

def normalizeRoIs(rois, imageDims):
    shape = imageDims['output_shape']

    norm_rois = np.zeros_like(rois)
    norm_rois[:,0] = rois[:,0] / (shape[0]-0)
    norm_rois[:,1] = rois[:,1] / (shape[1]-0)
    norm_rois[:,2] = rois[:,2] / (shape[0]-0)
    norm_rois[:,3] = rois[:,3] / (shape[1]-0)

    return norm_rois


def deltas2Boxes(props, deltas, rois, cfg, exclude_bg = True):
    nb_rois_in_batch = props.shape[1]
    bboxes = {}
    boxes = []
    
    for ii in range(nb_rois_in_batch):
#        if np.max(props[0, ii, :]) < bbox_threshold or np.argmax(props[0, ii, :]) == 0:
        if np.argmax(props[0, ii, :]) == 0 and exclude_bg:
            continue
        
        labelID = np.argmax(props[0, ii, :])
        if labelID not in bboxes:
            bboxes[labelID] = []
        prop = np.max(props[0, ii, :])
        roi = rois[ii, :]
        
        label_pos = labelID - 1
        
        regr = deltas[0, ii, 4 * label_pos:4 * (label_pos + 1)]
        sx, sy, sw, sh = cfg.det_regr_std
        regr[0] /= sx
        regr[1] /= sy
        regr[2] /= sw
        regr[3] /= sh
        x, y, w, h = apply_regr(roi, regr)
#        print('label', ii, labelID)
#        print('rt',roi[0], roi[1], roi[2], roi[3])
#        print('dl',regr[0], regr[1], regr[2], regr[3])
#        print('bb',x, y, w, h)
#        bboxes[labelID].append([x, y, (x + w), (y + h), prop])
        boxes.append([x, y, w, h, labelID, prop])

    boxes = np.array(boxes)
    return boxes

def non_max_suppression_boxes(bboxes, cfg, nms_overlap_thresh=0.5):
    # add some nms to reduce many boxes
    new_bboxes = []
    labelIDs = bboxes[:,4]
    for i in range(cfg.nb_object_classes):
        idxs = np.where(labelIDs == i)[0]
        sub_bboxes = bboxes[idxs,:]
        if sub_bboxes.shape[0] == 0:
            continue
        boxes_nms = non_max_suppression_fast(sub_bboxes, overlap_thresh=nms_overlap_thresh)
        new_bboxes.extend(boxes_nms)
        
    new_bboxes = np.array(new_bboxes)
    return new_bboxes

def deltas2Anchors(props, deltas, cfg, imageDims, do_regr=True):
    # Deltas to coordinates by way of anchors
    # [dx,dy,dw,dh] -> (xmin,ymin,width,height)
    
    assert props.shape[0] == 1
    shape = imageDims['redux_shape']
    deltas /= cfg.rpn_regr_std
    
    
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

            if do_regr:
                A[:, :, :, anc_idx] = apply_regr_np(A[:, :, :, anc_idx], regr)

            A[2, :, :, anc_idx] = np.maximum(1, A[2, :, :, anc_idx])
            A[3, :, :, anc_idx] = np.maximum(1, A[3, :, :, anc_idx])

            A[0, :, :, anc_idx] = np.maximum(0, A[0, :, :, anc_idx])
            A[1, :, :, anc_idx] = np.maximum(0, A[1, :, :, anc_idx])
            A[2, :, :, anc_idx] = np.minimum(shape[1] - 0.01 - A[0, :, :, anc_idx], A[2, :, :, anc_idx])
            A[3, :, :, anc_idx] = np.minimum(shape[0] - 0.01 - A[1, :, :, anc_idx], A[3, :, :, anc_idx])

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
    
    bg_samples = np.where(true_labels[0, :, 0] == 1)
    fg_samples = np.where(true_labels[0, :, 0] == 0)

    if len(bg_samples) > 0:
        bg_samples = bg_samples[0]
    else:
        bg_samples = []

    if len(fg_samples) > 0:
        fg_samples = fg_samples[0]
    else:
        fg_samples = []

    # Half positives, half negatives
    if len(fg_samples) < nb_detection_rois // 4:
        selected_pos_samples = fg_samples.tolist()
    else:
        selected_pos_samples = np.random.choice(fg_samples, nb_detection_rois // 4, replace=False).tolist()
    try:
        selected_neg_samples = np.random.choice(bg_samples, nb_detection_rois - len(selected_pos_samples),
                                                replace=False).tolist()
    except:
        selected_neg_samples = np.random.choice(bg_samples, nb_detection_rois - len(selected_pos_samples),
                                                replace=True).tolist()

    sel_samples = selected_pos_samples + selected_neg_samples
    return sel_samples



def getCOCOMapping():
    coco_mapping = {'bottle': 44, 'backpack': 27, 'handbag': 31, 'dog': 18, 'giraffe': 25, 'sink': 81, 'bench': 15, 'truck': 8, 'teddy bear': 88, 'book': 84, 'umbrella': 28, 'chair': 62, 'scissors': 87, 'toilet': 70, 'cat': 17, 'frisbee': 34, 'toothbrush': 90, 'oven': 79, 'baseball glove': 40, 'kite': 38, 'dining table': 67, 'parking meter': 14, 'bowl': 51, 'skis': 35, 'remote': 75, 'fire hydrant': 11, 'suitcase': 33, 'bird': 16, 'person': 1, 'zebra': 24, 'hair drier': 89, 'wine glass': 46, 'donut': 60, 'airplane': 5, 'elephant': 22, 'bus': 6, 'mouse': 74, 'boat': 9, 'tv': 72, 'horse': 19, 'car': 3, 'potted plant': 64, 'baseball bat': 39, 'train': 7, 'keyboard': 76, 'spoon': 50, 'tie': 32, 'motorcycle': 4, 'clock': 85, 'orange': 55, 'skateboard': 41, 'cup': 47, 'bed': 65, 'sandwich': 54, 'sports ball': 37, 'cake': 61, 'banana': 52, 'vase': 86, 'knife': 49, 'couch': 63, 'pizza': 59, 'cell phone': 77, 'stop sign': 13, 'microwave': 78, 'apple': 53, 'laptop': 73, 'carrot': 57, 'broccoli': 56, 'fork': 48, 'sheep': 20, 'cow': 21, 'hot dog': 58, 'surfboard': 42, 'tennis racket': 43, 'snowboard': 36, 'traffic light': 10, 'bicycle': 2, 'refrigerator': 82, 'bear': 23, 'toaster': 80}
    return coco_mapping

def getCOCOIDs(imagesMeta, class_mapping):
    coco_mapping = getCOCOMapping()
    
    category_ids = []
    img_ids = []
    
    for label in list(class_mapping.keys()):
        if label == 'bg':
            continue
        category_ids.append(coco_mapping[label])
    for imageID, _ in imagesMeta.items():
        img_ids.append(int(imageID))
    return img_ids, category_ids
    

def bboxes2COCOformat(bboxes, imageMeta, class_mapping, scale=[1,1], rpn_stride=1, shape=[1,1], roundoff=False):
    results = []
    coco_mapping = getCOCOMapping()
    inv_class_mapping = {idx:label for label,idx in class_mapping.items()}
    for bbox in bboxes:
        label = bbox[4]
        label = inv_class_mapping[label]
        label = coco_mapping[label]
        prop = bbox[5]
        coords = bbox[:4]
        xmin = ((coords[0]) * rpn_stride / scale[0])
        ymin = ((coords[1]) * rpn_stride / scale[1])
        width = ((coords[2]) *  rpn_stride / scale[0])
        height = ((coords[3]) * rpn_stride / scale[1])
        coords = [xmin, ymin, width, height]
        coords = [round(float(x),2) for x in coords]
        
        res = {'image_id': int(imageMeta['id']), 'category_id': int(label), 'bbox': coords, 'score': round(float(prop),4)}
#        coords.append(prop)
#        res = coords
        results.append(res)
#    results = np.array(results)
    return results

def bboxes2HOIformat(h_bboxes, o_bboxes, hoi_labels, val_map):
    hoi_labels = [np.where(x==1)[0].tolist() for x in hoi_labels.astype(int)]
    h_bboxes = [[round(float(x), 2) for x in box] for box in h_bboxes.tolist()]
    o_bboxes = [[round(float(x), 2) for x in box] for box in o_bboxes.tolist()]
    val_map = val_map.astype(int).tolist()
    return h_bboxes, o_bboxes, hoi_labels, val_map


def bboxes2DETformat(bboxes, target_labels, target_deltas, cfg):
    target_labels = [int(np.argmax(x)) for x in target_labels]
    bboxes = [[round(float(x), 4) for x in box] for box in bboxes.tolist()]
    target_deltas = [[round(float(row[x]), 4) for x in np.where(row!=0)[0].tolist()] for row in target_deltas[:,(cfg.nb_object_classes-1)*4:]]
    return bboxes, target_labels, target_deltas



def _computeIoUs(bbox, gt_bboxes):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    gt_areas = (gt_bboxes[:,2] - gt_bboxes[:,0]) * (gt_bboxes[:,3] - gt_bboxes[:,1])
    
    # find the intersection
    xx1_int = np.maximum(bbox[0], gt_bboxes[:,0])
    yy1_int = np.maximum(bbox[1], gt_bboxes[:,1])
    xx2_int = np.minimum(bbox[2], gt_bboxes[:,2])
    yy2_int = np.minimum(bbox[3], gt_bboxes[:,3])

    ww_int = np.maximum(0, xx2_int - xx1_int)
    hh_int = np.maximum(0, yy2_int - yy1_int)

    area_int = ww_int * hh_int
    # find the union
    area_union = area + gt_areas - area_int

    # compute the ratio of overlap
    overlap = area_int / (area_union + 1e-6)
    
    return overlap

def _transformBBoxes(bboxes, dosplit=True):
    hbboxes = []
    obboxes = []
    gt_allboxes = []

    
    bboxes = bboxes[bboxes[:,5].argsort()[::-1]]
    
    for bbox in bboxes:
        label = bbox[4]
        prop = bbox[5]
        (xmin, ymin, width, height) = bbox[:4]
        xmax = xmin + width
        ymax = ymin + height
        trans_bbox = [xmin, ymin, xmax, ymax, label, prop]
        
        if not dosplit:
            gt_allboxes.append(trans_bbox)
        if label == 1: #human
            hbboxes.append(trans_bbox)
        elif label > 1: #object
            obboxes.append(trans_bbox)
            
    if not dosplit:
        return np.array(gt_allboxes)

    if len(hbboxes)==0 or len(obboxes)==0:
        return None, None
    
    obboxes = np.array(obboxes + cp.copy(hbboxes))
    hbboxes = np.array(hbboxes)

    return hbboxes, obboxes   


def _transformGTBBox(gt_bboxes, class_mapping, gt_rels, scale=[1,1], rpn_stride=1, shape=[1,1], roundoff=False, dosplit=True):
    gt_hbboxes = []
    gt_obboxes = []
    gt_allboxes = []
    
    h_idxs = gt_rels[:,0]
    o_idxs = gt_rels[:,1]

    
    for b_idx, gt_bbox in enumerate(gt_bboxes):
        label = gt_bbox['label']
        label = class_mapping[label]
        xmin = ((gt_bbox['xmin']) * scale[0] / rpn_stride) / shape[0]
        xmax = ((gt_bbox['xmax']-0.01) * scale[0] / rpn_stride) / shape[0]
        ymin = ((gt_bbox['ymin']) * scale[1] / rpn_stride) / shape[1]
        ymax = ((gt_bbox['ymax']-0.01) * scale[1] / rpn_stride) / shape[1]
        if roundoff:
            xmin=int(round(xmin)); xmax=int(round(xmax))
            ymin=int(round(ymin)); ymax=int(round(ymax))
            
        trans_gt_bbox = [xmin, ymin, xmax, ymax, label]
        
        if not dosplit:
            gt_allboxes.append(trans_gt_bbox)
        if b_idx in h_idxs: #human
            gt_hbboxes.append(trans_gt_bbox)
        elif b_idx in o_idxs: #object
            gt_obboxes.append(trans_gt_bbox)
        
        
    if not dosplit:
        return np.array(gt_allboxes)
        
    gt_hbboxes = np.array(gt_hbboxes)
    gt_obboxes = np.array(gt_obboxes)

    return gt_hbboxes, gt_obboxes   


def _getRelMap(rels):
    nb_prs = len(np.unique(rels[:,0]))
    nb_obj = len(np.unique(rels[:,1]))
    
    prsidxs = np.unique(rels[:,0])
    prsidxs = {idx:i for i,idx in enumerate(prsidxs)}
    
    objidxs = np.unique(rels[:,1])
    objidxs = {idx:i for i,idx in enumerate(objidxs)}
    
    rel_map = np.ones((nb_prs, nb_obj)) * -1
    
    for rel in rels:
        rel_map[prsidxs[rel[0]], objidxs[rel[1]]] = rel[2]
    return rel_map


       
def preprocessImage(img, cfg):
    img = img.astype(np.float32, copy=False)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img /= 127.5
    img -= 1.0
    
    img_shape = img.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])
    img_scale = float(cfg.mindim) / float(img_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(img_scale * img_size_min) > cfg.maxdim:
        img_scale = float(cfg.maxdim) / float(img_size_max)
    img = cv.resize(img, None, None, fx=img_scale, fy=img_scale,
                    interpolation=cv.INTER_LINEAR)
    return img, [img_scale, img_scale]

       
def normalizeGTboxes(gtboxes, scale=[1,1], rpn_stride=1, shape=[1,1], roundoff=False):
    gtnormboxes = []
    for relID, bbox in enumerate(gtboxes):
#        print(bbox)
        # get the GT box coordinates, and resize to account for image resizing
        xmin = ((bbox['xmin']) * scale[0] / rpn_stride) / shape[0]
        xmax = ((bbox['xmax']-0.01) * scale[0] / rpn_stride) / shape[0]
        ymin = ((bbox['ymin']) * scale[1] / rpn_stride) / shape[1]
        ymax = ((bbox['ymax']-0.01) * scale[1] / rpn_stride) / shape[1]
        if roundoff:
            xmin=int(round(xmin)); xmax=int(round(xmax))
            ymin=int(round(ymin)); ymax=int(round(ymax))
        gtnormboxes.append({'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax})    
    return gtnormboxes

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
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.hoi_neg2_share, replace=False)
    elif len(negative2_idxs) > 0:
        selected_neg2_samples = np.random.choice(negative2_idxs, cfg.nb_hoi_rois - len(selected_pos_samples) - len(selected_neg1_samples), replace=True)
    else:
        selected_neg1_samples = np.random.choice(negative1_idxs, cfg.nb_hoi_rois - len(selected_pos_samples), replace=True)
        

    selected_samples = selected_pos_samples.tolist() + selected_neg1_samples.tolist() + selected_neg2_samples.tolist()
    return selected_samples

