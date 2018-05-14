# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:37:06 2018

@author: aag14
"""
import numpy as np
import copy
import utils

import filters_helper as helper

def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

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
        return X


def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300):
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    # TODO: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob] format!!!! with prob built in
    if len(boxes) == 0:
        return []
    # normalize to np.array
    boxes = np.array(boxes)
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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


def deltas_to_roi(rpn_layer, regr_layer, cfg):
#    regr_layer = regr_layer / cfg.std_scaling
    
    #############################
    ######## Parameters #########
    #############################    
    max_boxes=cfg.nms_max_boxes
    overlap_thresh=cfg.nms_overlap_tresh

    anchor_sizes = cfg.anchor_sizes
    anchor_ratios = cfg.anchor_ratios

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    anc_idx = 0
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            
            anchor_x = (anchor_size * anchor_ratio[0]) / cfg.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / cfg.rpn_stride
            
            regr = regr_layer[0, :, :, 4 * anc_idx:4 * anc_idx + 4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            
            A[0, :, :, anc_idx] = X - anchor_x / 2
            A[1, :, :, anc_idx] = Y - anchor_y / 2
            A[2, :, :, anc_idx] = anchor_x
            A[3, :, :, anc_idx] = anchor_y
            
            A[:, :, :, anc_idx] = apply_regr_np(A[:, :, :, anc_idx], regr)

            A[2, :, :, anc_idx] = np.maximum(1, A[2, :, :, anc_idx])
            A[3, :, :, anc_idx] = np.maximum(1, A[3, :, :, anc_idx])
            A[2, :, :, anc_idx] += A[0, :, :, anc_idx]
            A[3, :, :, anc_idx] += A[1, :, :, anc_idx]

            A[0, :, :, anc_idx] = np.maximum(0, A[0, :, :, anc_idx])
            A[1, :, :, anc_idx] = np.maximum(0, A[1, :, :, anc_idx])
            A[2, :, :, anc_idx] = np.minimum(cols - 1, A[2, :, :, anc_idx])
            A[3, :, :, anc_idx] = np.minimum(rows - 1, A[3, :, :, anc_idx])

            anc_idx += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, ids, 0)
    all_probs = np.delete(all_probs, ids, 0)

    # I guess boxes and prob are all 2d array, I will concat them
    all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs])))
    result = non_max_suppression_fast(all_boxes, overlap_thresh=overlap_thresh, max_boxes=max_boxes)
    # omit the last column which is prob
    result = result[:, 0: -1]
    return result


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



def detection_ground_truths(rois, imageMeta, imageDims, class_mapping, cfg):    
    #############################
    ########## Image ############
    #############################
    bboxes = imageMeta['objects']
    
    scale = imageDims['scale']
    shape = imageDims['shape']

    #############################
    ###### Set Parameters #######
    #############################    
    rpn_stride             = cfg.rpn_stride
    detection_max_overlap = cfg.detection_max_overlap
    detection_min_overlap = cfg.detection_min_overlap
    
    #############################
    #### Initialize matrices ####
    #############################    
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []  # for debugging only

    #############################
    ##### Ground truth boxes ####
    #############################
    gta = helper.normalizeGTboxes(bboxes, scale=scale, rpn_stride=rpn_stride)


    #############################
    #### Ground truth objects ###
    #############################
    for ix in range(rois.shape[0]):
        (xmin, ymin, xmax, ymax) = rois[ix, :]
        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))
        
        rt = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

        best_iou = 0.0
        best_bbox = -1
        for bbidx, gt in enumerate(gta):
            curr_iou = utils.get_iou(gt, rt)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbidx

        if best_iou < detection_min_overlap:
            continue
        else:
            wr = rt['xmax'] - rt['xmin']
            hr = rt['ymax'] - rt['ymin']
            x_roi.append([rt['ymin']/shape[0], rt['xmin']/shape[1], rt['ymax']/shape[0], rt['xmax']/shape[1]])
            IoUs.append(best_iou)

            if detection_min_overlap <= best_iou < detection_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif detection_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['label']
                cxg = (gt['xmin'] + gt['xmax']) / 2.0
                cyg = (gt['ymin'] + gt['ymax']) / 2.0

                cxr = (rt['xmin'] + rt['xmax']) / 2.0
                cyr = (rt['ymin'] + rt['ymax']) / 2.0
                
                wg = gt['xmax'] - gt['xmin']
                hg = gt['ymax'] - gt['ymin']

                tx = (cxg - cxr) / float(wr)
                ty = (cyg - cyr) / float(hr)
                tw = np.log(wg / float(wr))
                th = np.log(hg / float(hr))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # Classification ground truth
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        
        # Regression ground truth
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
#            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4 + label_pos] = [tx, ty, tw, th]
#            coords[label_pos+0] * sx; coords[label_pos+1] * sy
#            coords[label_pos+2] * sw; coords[label_pos+3] * sh
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    rois = np.array(x_roi)
    rois = np.insert(rois, 0, 0, axis=1)    
    
    true_labels = np.array(y_class_num)
    true_boxes = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(rois, axis=0), np.expand_dims(true_labels, axis=0), np.expand_dims(true_boxes, axis=0), IoUs
