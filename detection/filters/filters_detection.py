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
    rois = rois[0,:,1:]
    rois = rois[:,(1,0,3,2)]

    rois[:,2] = rois[:,2] - rois[:,0]
    rois[:,3] = rois[:,3] - rois[:,1]
    
    rois = helper.unnormalizeRoIs(rois, imageDims)
    return rois

def prepareInputs(rois, imageDims):
    #in: bboxes of (xmin,ymin,width,height) in range [0,output_shape]
    #out: bboxes of (idx,ymin,xmin,ymax,xmax) in range [0,1]
    
    new_rois = np.copy(rois)
    new_rois = new_rois[:,(1,0,3,2)]
    new_rois[:,2] = new_rois[:,2] + new_rois[:,0]
    new_rois[:,3] = new_rois[:,3] + new_rois[:,1]
    
    new_rois = helper.normalizeRoIs(new_rois, imageDims)
    
    new_rois = np.insert(new_rois, 0, 0, axis=1)
    new_rois = np.expand_dims(new_rois, axis=0)
    
    return new_rois

#######################
### PROCESS TARGETS ###
#######################
def loadTargets(imageMeta, rois_path, imageDims, cfg, batchidx = None):
    roisMeta = utils.load_dict(rois_path + str(int(imageMeta['imageName'].split('.')[0])))
    if roisMeta is None:
        return None, None, None
    allbboxes = np.array(roisMeta['proposals'])
    alltarget_labels = np.array(roisMeta['target_labels'])
    alltarget_deltas = np.array(roisMeta['target_deltas'])
    
    alltarget_deltas = utils.getMatrixDeltas(cfg.nb_object_classes, alltarget_deltas, alltarget_labels)
    alltarget_labels = utils.getMatrixLabels(cfg.nb_object_classes, alltarget_labels)

    allbboxes = prepareInputs(allbboxes, imageDims)    
    alltarget_labels = np.expand_dims(alltarget_labels, axis=0)
    alltarget_deltas = np.expand_dims(alltarget_deltas, axis=0)
    
    rois_redux, target_props_redux, target_deltas_redux = reduceTargets(allbboxes, alltarget_labels, alltarget_deltas, cfg)
        
    
    return rois_redux, target_props_redux, target_deltas_redux


def reduceTargets(bboxes, target_labels, target_deltas, cfg, batchidx=None):
    ## Pick reduced indexes ##
    if batchidx is None:        
        nb_detection_rois = cfg.nb_detection_rois
        bg_samples = np.where(target_labels[:, 0] == 1)
        fg_samples = np.where(target_labels[:, 0] == 0)
    
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
    else:
        bboxes = np.zeros((cfg.nb_detection_rois, 5))
        target_props = np.zeros((cfg.nb_detection_rois, cfg.nb_object_classes))
        target_deltas = np.zeros((cfg.nb_detection_rois, (cfg.nb_object_classes-1)*4*2))
        
        sidx = batchidx * cfg.nb_detection_rois
        fidx = min(cfg.detection_nms_max_boxes, sidx + cfg.nb_detection_rois)
        sel_samples = list(range(sidx,fidx))
        
    
    ## Reduce data by picked indexes ##  
    bboxes = bboxes[sel_samples, :]
    target_props = target_labels[sel_samples, :]
    target_deltas = target_deltas[sel_samples, :]
    
    return bboxes, target_props, target_deltas
    


def createTargets(bboxes, imageMeta, imageDims, class_mapping, cfg):    
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
        (xmin, ymin, width, height) = bboxes[ix, :]
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

