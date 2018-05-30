# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:37:06 2018

@author: aag14
"""
import numpy as np
import copy
import utils

import filters_helper as helper

def loadData(imageMeta, rois_path, cfg, batchidx = None):
    roisMeta = utils.load_dict(rois_path + imageMeta['imageName'].split('.')[0])
#    roisMeta = utils.load_dict(rois_path + imageMeta['imageName'])
    if roisMeta is None:
        return None, None, None
    allrois = np.array(roisMeta['rois'])
    alltarget_props = np.array(roisMeta['target_props'])
    alltarget_deltas = np.array(roisMeta['target_deltas'])
    
    if batchidx is None:
        samples = helper.reduce_rois(alltarget_props, cfg)
        rois = allrois[:,samples, :]
        target_props = alltarget_props[:, samples, :]
        target_deltas = alltarget_deltas[:, samples, :]
    else:
        rois = np.zeros((1, cfg.nb_detection_rois, 5))
        target_props = np.zeros((1, cfg.nb_detection_rois, cfg.nb_object_classes))
        target_deltas = np.zeros((1, cfg.nb_detection_rois, (cfg.nb_object_classes-1)*4*2))
        
        sidx = batchidx * cfg.nb_detection_rois
        fidx = min(cfg.detection_nms_max_boxes, sidx + cfg.nb_detection_rois)
        rois[:,:fidx-sidx,:] = allrois[:,sidx:fidx, :]
        target_props[:,:fidx-sidx,:] = alltarget_props[:, sidx:fidx, :]
        target_deltas[:,:fidx-sidx,:] = alltarget_deltas[:, sidx:fidx, :]
    
    return rois, target_props, target_deltas

def unprepareInputs(norm_rois, imageDims):
    #(idx,ymin,xmin,ymax,xmax) -> (xmin,ymin,width,height)
    
    rois = copy.copy(norm_rois)
    rois = rois[0,:,1:]
    rois = rois[:,(1,0,3,2)]

    rois[:,2] = rois[:,2] - rois[:,0]
    rois[:,3] = rois[:,3] - rois[:,1]
    
    rois = helper.unnormalizeRoIs(rois, imageDims)
    return rois

def prepareInputs(rois, imageDims):
    #(xmin,ymin,width,height) -> (idx,ymin,xmin,ymax,xmax)
    
#    print(rois.shape)
    new_rois = copy.copy(rois)
    new_rois = new_rois[:,(1,0,3,2)]
    new_rois[:,2] = new_rois[:,2] + new_rois[:,0]
    new_rois[:,3] = new_rois[:,3] + new_rois[:,1]
    
    new_rois = helper.normalizeRoIs(new_rois, imageDims)
    
    new_rois = np.insert(new_rois, 0, 0, axis=1)
    new_rois = np.expand_dims(new_rois, axis=0)
    
    return new_rois


def prepareTargets(rois, imageMeta, imageDims, class_mapping, cfg):    
    #############################
    ########## Image ############
    #############################
    bboxes = imageMeta['objects']
    
    scale = imageDims['scale']
#    shape = imageDims['shape']

    #############################
    ###### Set Parameters #######
    #############################    
    rpn_stride            = cfg.rpn_stride
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
        (xmin, ymin, width, height) = rois[ix, :]
#        xmin = int(round(xmin))
#        ymin = int(round(ymin))
#        xmax = int(round(xmax))
#        ymax = int(round(ymax))
        
        rt = {'xmin': xmin, 'ymin': ymin, 'xmax': xmin+width, 'ymax': ymin+height}

        best_iou = 0.0
        best_bbox = -1
        for bbidx, gt in enumerate(gta):
            curr_iou = utils.get_iou(gt, rt)
            if curr_iou > best_iou:
#                print(curr_iou)
                best_iou = curr_iou
                best_bbox = bbidx

        if best_iou < detection_min_overlap:
            continue
        else:
            x_roi.append([xmin, ymin, width, height])
            IoUs.append(best_iou)

            if detection_min_overlap <= best_iou < detection_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif detection_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['label']                
                tx, ty, tw, th = helper.get_GT_deltas(gta[best_bbox], rt)
#                bxmin, bymin, bw, bh = helper.apply_regr([xmin,ymin,width,height], [tx,ty,tw,th])
#                print(best_iou)
#                print('rt',rt['xmin'], rt['ymin'], rt['xmax'], rt['ymax'])
#                print('gt',gta[best_bbox]['xmin'], gta[best_bbox]['ymin'], gta[best_bbox]['xmax'], gta[best_bbox]['ymax'])
#                print('bb',bxmin, bymin, bxmin + bw, bymin + bh)
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
            label_pos = 4 * (class_num - 1)
            sx, sy, sw, sh = cfg.det_regr_std
            coords[label_pos:4 + label_pos] = [tx*sx, ty*sy, tw*sw, th*sh]
#            coords[label_pos+0] * sx; coords[label_pos+1] * sy
#            coords[label_pos+2] * sw; coords[label_pos+3] * sh
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
#        print('x roi none')
        return None, None, None, None

    rois = np.array(x_roi)
    y_class_regr_label = np.array(y_class_regr_label)
    y_class_regr_coords = np.array(y_class_regr_coords)
    
    true_labels = np.array(y_class_num)
    true_boxes = np.concatenate([y_class_regr_label, y_class_regr_coords], axis=1)

    return rois, np.expand_dims(true_labels, axis=0), np.expand_dims(true_boxes, axis=0), IoUs


########################
###### OUT DATED #######
########################

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
    # regr_layer = regr_layer / cfg.std_scaling
    # [dx,dy,dw,dh] -> (xmin,ymin,xmax,ymax)
    
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