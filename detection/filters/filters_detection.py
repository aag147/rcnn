# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:37:06 2018

@author: aag14
"""
import numpy as np
import copy
import utils

import filters_helper as helper


#######################
#### PROCESS INPUT ####
#######################
def unprepareInputs(norm_rois, imageDims):
    #in: bboxes of (idx,ymin,xmin,ymax,xmax) in range [0,1]
    #out: bboxes of (xmin,ymin,width,height) in range [0,output_shape]
    
    rois = np.copy(norm_rois)
    rois = rois[:,:,1:]
    rois = rois[:,:,(1,0,3,2)]

    rois[:,:,2] = rois[:,:,2] - rois[:,:,0]
    rois[:,:,3] = rois[:,:,3] - rois[:,:,1]
    
    rois = helper.unnormalizeRoIs(rois, imageDims)
    
    return rois

def prepareInputs(rois, imageDims, imageMeta=None):
    #in: bboxes of (xmin,ymin,width,height) in range [0,output_shape]
    #out: bboxes of (idx,ymin,xmin,ymax,xmax) in range [0,1]
    
    new_rois = np.copy(rois)
    new_rois = new_rois[:,:,(1,0,3,2)]
    new_rois[:,:,2] = new_rois[:,:,2] + new_rois[:,:,0]
    new_rois[:,:,3] = new_rois[:,:,3] + new_rois[:,:,1]
    
    new_rois = helper.normalizeRoIs(new_rois, imageDims)
    new_rois = np.insert(new_rois, 0, 0, axis=2)
    
    imageID = imageMeta['imageID']
    assert(np.all(new_rois[0,:,1]>=0))
    assert(np.all(new_rois[0,:,2]>=0))
    assert(np.all(new_rois[0,:,3]<=1.0))
    assert(np.all(new_rois[0,:,4]<=1.0))
    
    try:
        np.testing.assert_array_less(new_rois[0,:,1], new_rois[0,:,3], err_msg='imageID: '+str(imageID))
        np.testing.assert_array_less(new_rois[0,:,2], new_rois[0,:,4], err_msg='imageID: '+str(imageID))
    except AssertionError:
        print('bad imageID', str(imageID))
    
    return new_rois

#######################
##### MANAGE DATA #####
#######################
def loadData(imageInputs, cfg):
#def loadData(imageMeta, rois_path, imageDims, cfg, batchidx = None):
    #out: rois [{1}, {...}, (1,ymin,xmin,ymax,xmax)]
    #out: labels [{1}, {...}, {nb_object_classes}]
    #out: deltas [{1}, {...}, (dx,dy,dw,dh) * (nb_object_classes-1)]

    roisMeta = imageInputs    
#    roisMeta = utils.load_obj(rois_path + imageMeta['imageName'].split('.')[0])    

    if roisMeta is None:
        return None
    all_bboxes = np.array(roisMeta['rois']).astype(np.float64)
    all_bboxes /= 1000.0
    all_target_labels = np.array(roisMeta['target_props'])
    all_target_deltas = roisMeta['target_deltas']
    
    all_target_deltas = utils.getMatrixDeltas(cfg.nb_object_classes, all_target_deltas, all_target_labels).astype(np.float64)
    all_target_deltas[:,(cfg.nb_object_classes-1)*4:] /= 1000.0
    all_target_labels = utils.getMatrixLabels(cfg.nb_object_classes, all_target_labels)

    all_bboxes = np.expand_dims(all_bboxes, axis=0)
    all_target_labels = np.expand_dims(all_target_labels, axis=0)
    all_target_deltas = np.expand_dims(all_target_deltas, axis=0)    
    
    return [all_bboxes, all_target_labels, all_target_deltas]


def convertData(Y, cfg):
    [all_bboxes, all_target_labels, all_target_deltas] = Y
    
    all_bboxes = np.copy(all_bboxes[0])
    all_target_labels = np.copy(all_target_labels[0])
    all_target_deltas = np.copy(all_target_deltas[0])
    
    all_target_labels = [int(np.argmax(x)) for x in all_target_labels]
#    all_bboxes = [[round(float(x), 4) for x in box] for box in all_bboxes.tolist()]
    all_bboxes = [[int(x*1000) for x in box] for box in all_bboxes.tolist()]
    new_target_deltas = []
    for idx, row in enumerate(all_target_deltas[:,(cfg.nb_object_classes-1)*4:]):
        coord = []
        for x in row[(all_target_labels[idx]-1)*4:(all_target_labels[idx])*4].tolist():
#            coord.append(round(float(x), 4))
            coord.append(int(x*1000))
        new_target_deltas.append(coord)
        
        
    detMeta = {'rois':all_bboxes, 'target_props':all_target_labels, 'target_deltas':new_target_deltas}
    return detMeta


def convertResults(bboxes, imageMeta, class_mapping, scale, rpn_stride):
    bboxes = np.copy(bboxes)
    results = []
    coco_mapping = helper.getCOCOMapping()
    inv_class_mapping = {idx:label for label,idx in class_mapping.items()}
    for bbox in bboxes:
        label = bbox[5]
        label = inv_class_mapping[label]
        label = coco_mapping[label]
        prop = bbox[4]
        coords = bbox[:4]
        xmin = ((coords[0]) * rpn_stride / scale[0])
        ymin = ((coords[1]) * rpn_stride / scale[1])
        width = ((coords[2]) *  rpn_stride / scale[0])
        height = ((coords[3]) * rpn_stride / scale[1])
        coords = [xmin, ymin, width, height]
        coords = [round(float(x),2) for x in coords]
        
        res = {'image_id': int(imageMeta['imageID']), 'category_id': int(label), 'bbox': coords, 'score': round(float(prop),4)}
        results.append(res)
    return results
        
def reduceData(Y, cfg, batchidx=None):
    #out: bboxes [{1}, {batch_size}, (0,ymin,xmin,ymax,xmax)]
    #out: labels [{1}, {batch_size}, {nb_object_classes}]
    #out: deltas [{1}, {batch_size}, (dx,dy,dw,dh) * (nb_object_classes-1)]
    [all_bboxes, all_target_labels, all_target_deltas] = Y
    all_bboxes = all_bboxes[:,:,:4]
    
    
    bboxes = np.zeros((1, cfg.nb_detection_rois, 4))
    target_labels = np.zeros((1, cfg.nb_detection_rois, cfg.nb_object_classes))
    target_deltas = np.zeros((1, cfg.nb_detection_rois, (cfg.nb_object_classes-1)*4*2))
    
    ## Pick reduced indexes ##
    if batchidx is None:        
        bg_samples = np.where(all_target_labels[0,:, 0] == 1)
        fg_samples = np.where(all_target_labels[0,:, 0] == 0)
    
        if len(bg_samples) > 0:
            bg_samples = bg_samples[0]
        else:
            bg_samples = []
    
        if len(fg_samples) > 0:
            fg_samples = fg_samples[0]
        else:
            fg_samples = []
    
        # Half positives, half negatives
        nb_max_fg = int(cfg.nb_detection_rois * cfg.det_fg_ratio)
        if len(fg_samples) == 0:
            selected_pos_samples = []  
        elif len(fg_samples) < nb_max_fg:
            selected_pos_samples = fg_samples.tolist()
#            selected_pos_samples = np.random.choice(fg_samples, nb_max_fg, replace=True).tolist()
        else:
            selected_pos_samples = np.random.choice(fg_samples, nb_max_fg, replace=False).tolist()
        try:
            selected_neg_samples = np.random.choice(bg_samples, cfg.nb_detection_rois - len(selected_pos_samples),
                                                    replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(bg_samples, cfg.nb_detection_rois - len(selected_pos_samples),
                                                    replace=True).tolist()
    
        sel_samples = selected_pos_samples + selected_neg_samples
    else:        
        sidx = batchidx * cfg.nb_detection_rois
        fidx = min(cfg.detection_nms_max_boxes, sidx + cfg.nb_detection_rois)
        sel_samples = list(range(sidx,fidx))
        
    
    assert(target_labels.shape[1] == cfg.nb_detection_rois)
    
    ## Reduce data by picked indexes ##  
    bboxes[:,:len(sel_samples),:]           = all_bboxes[:, sel_samples, :]
    target_labels[:,:len(sel_samples),:]    = all_target_labels[:, sel_samples, :]
    target_deltas[:,:len(sel_samples),:]    = all_target_deltas[:, sel_samples, :]
    
    return bboxes, target_labels, target_deltas
    

#######################
### PROCESS TARGETS ###
#######################
def createTargets(bboxes, imageMeta, imageDims, class_mapping, cfg):
    #out: rois [{1}, {...}, (1,ymin,xmin,ymax,xmax)]
    #out: labels [{1}, {...}, {nb_object_classes}]
    #out: deltas [{1}, {...}, (dx,dy,dw,dh) * (nb_object_classes-1)]
    
    #############################
    ########## Image ############
    #############################
    gt_bboxes = imageMeta['objects']
    
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
    gta = helper.normalizeGTboxes(gt_bboxes, scale=scale, rpn_stride=rpn_stride)

    #############################
    #### Ground truth objects ###
    #############################
    for ix in range(bboxes.shape[0]):
        (xmin, ymin, width, height, prop) = bboxes[ix, :5]
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
            x_roi.append([xmin, ymin, width, height, prop])
            IoUs.append(best_iou)

            if detection_min_overlap <= best_iou < detection_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif detection_max_overlap <= best_iou:
                cls_name = gt_bboxes[best_bbox]['label']                
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

    return np.expand_dims(rois, axis=0), np.expand_dims(true_labels, axis=0), np.expand_dims(true_boxes, axis=0), IoUs

