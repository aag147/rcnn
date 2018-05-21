# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:44:44 2018

@author: aag14
"""

import cv2 as cv
import numpy as np
import random
import filters_helper as helper
import utils


def prepareInputs(imageMeta, images_path, cfg):
    img = cv.imread(images_path + imageMeta['imageName'])
#    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    assert(img is not None)
    assert(img.shape[0] > 10)
    assert(img.shape[1] > 10)
    assert(img.shape[2] == 3)
    imgRedux, scale = helper.preprocessImage(img, cfg)
    output_shape = [imgRedux.shape[0] / cfg.rpn_stride, imgRedux.shape[1] / cfg.rpn_stride]
    imgDims = {'shape': img.shape, 'redux_shape':imgRedux.shape, 'output_shape':output_shape, 'scale':scale}
    imgRedux = np.expand_dims(imgRedux, axis=0)
    return imgRedux, imgDims


def loadTargets(imageMeta, anchors_path):
    y_rpn_cls, y_rpn_regr = utils.load_obj(anchors_path + imageMeta['imageName'].split('.')[0])
    return y_rpn_cls, y_rpn_regr

def prepareTargets(imageMeta, imageDims, cfg):

    #############################
    ########## Image ############
    #############################
    bboxes = imageMeta['objects']
    scale = imageDims['scale']
    reduced_shape = imageDims['redux_shape']
    image_height = reduced_shape[0]
    image_width  = reduced_shape[1]
    
    #############################
    ###### Set Parameters #######
    #############################
    rpn_stride = cfg.rpn_stride
    
    output_width = int(image_width / rpn_stride)
    output_height = int(image_height / rpn_stride)
    
    anchor_sizes = cfg.anchor_sizes
    anchor_ratios = cfg.anchor_ratios
    
    num_anchors = len(anchor_sizes) * len(anchor_ratios)
    
    rpn_min_overlap = cfg.rpn_min_overlap
    rpn_max_overlap = cfg.rpn_max_overlap
    
    
    #############################
    #### Initialize matrices ####
    #############################
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))
    
    num_bboxes = len(bboxes)
    
    num_anchors_for_gtbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_gtbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_gtbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_gtbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_gtbox = np.zeros((num_bboxes, 4)).astype(np.float32)
    
    #############################
    ##### Ground truth boxes ####
    #############################
    gta = helper.normalizeGTboxes(bboxes, scale=scale, roundoff=True)
#    draw.drawHOI(image, gta[0,:], gta[0,:])
     
    #############################
    # Map ground truth 2 anchor #
    #############################
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(len(anchor_ratios)):
            w_anc = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            h_anc = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
            
            for ix in range(output_width):
                xmin_anc = float(rpn_stride) * (ix + 0.5) - w_anc / 2
                xmax_anc = float(rpn_stride) * (ix + 0.5) + w_anc / 2
                if xmin_anc < 0 or xmax_anc > image_width:
                    continue                
                
                for jy in range(output_height):
                    ymin_anc = float(rpn_stride) * (jy + 0.5) - h_anc / 2
                    ymax_anc = float(rpn_stride) * (jy + 0.5) + h_anc / 2
                    if ymin_anc < 0 or ymax_anc > image_height:
                        continue
                    
                    bbox_type = 'neg'
                    best_iou_for_loc = 0.0
                    at = {'xmin': xmin_anc, 'ymin': ymin_anc, 'xmax': xmax_anc, 'ymax': ymax_anc}
    #                print((rpn_stride*(ix+0.5), rpn_stride*(jy+0.5)), anchor_sizes[anchor_size_idx], anchor_ratios[anchor_ratio_idx])
                    for gtidx in range(num_bboxes):
                        gt = gta[gtidx]
                        curr_iou = utils.get_iou(gt, at)
                                                
                        if curr_iou > best_iou_for_gtbox[gtidx] or curr_iou > rpn_max_overlap:
                            tx, ty, tw, th = helper.get_GT_deltas(gt, at)
                            
#                            bxmin, bymin, bw, bh = helper.apply_regr([at['xmin'],at['ymin'],at['xmax']-at['xmin'],at['ymax']-at['ymin']], [tx,ty,tw,th])
#                            print(curr_iou)
#                            print('at',at['xmin'], at['ymin'], at['xmax'], at['ymax'])
#                            print('gt',gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax'])
#                            print('bb',bxmin, bymin, bxmin + bw, bymin + bh)
    						
    					# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                        if curr_iou > best_iou_for_gtbox[gtidx]:
                            best_anchor_for_gtbox[gtidx] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                            best_iou_for_gtbox[gtidx] = curr_iou
                            best_x_for_gtbox[gtidx,:] = [at['xmin'], at['xmax'], at['ymin'], at['ymax']]
                            best_dx_for_gtbox[gtidx,:] = [tx, ty, tw, th]
    
    					# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                        if curr_iou > rpn_max_overlap:
#                            print(curr_iou, at)
    #                        print(anchor_sizes[anchor_size_idx], anchor_ratios[anchor_ratio_idx])
                            bbox_type = 'pos'
                            num_anchors_for_gtbox[gtidx] += 1
    						# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                            if curr_iou > best_iou_for_loc:
                                best_iou_for_loc = curr_iou
                                best_regr = (tx, ty, tw, th)
    
    					# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                        if rpn_min_overlap < curr_iou < rpn_max_overlap:
    						# gray zone between neg and pos
                            if bbox_type != 'pos':
                                bbox_type = 'neutral'
    
    				# turn on or off outputs depending on IOUs
                    anc_idx = (anchor_ratio_idx + len(anchor_ratios) * anchor_size_idx)
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anc_idx] = 1
                        y_rpn_overlap[jy, ix, anc_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anc_idx] = 0
                        y_rpn_overlap[jy, ix, anc_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anc_idx] = 1
                        y_rpn_overlap[jy, ix, anc_idx] = 1
                        y_rpn_regr[jy, ix, 4*anc_idx:4*anc_idx+4] = best_regr
                        
            
    #############################
    ##### Ensure GT Anchors #####
    #############################        
    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_gtbox.shape[0]):
        if num_anchors_for_gtbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_gtbox[idx, 0] == -1:
                continue
            
            anc_idx = best_anchor_for_gtbox[idx,2] + len(anchor_ratios) * best_anchor_for_gtbox[idx,3]
            y_is_box_valid[
                best_anchor_for_gtbox[idx,0], best_anchor_for_gtbox[idx,1], anc_idx] = 1
            y_rpn_overlap[
                best_anchor_for_gtbox[idx,0], best_anchor_for_gtbox[idx,1], anc_idx] = 1
            y_rpn_regr[
                best_anchor_for_gtbox[idx,0], best_anchor_for_gtbox[idx,1], 4*anc_idx:4*anc_idx+4] = best_dx_for_gtbox[idx, :]
    
#    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
    
#    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
    
#    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
    
    #############################
    ##### Reduce GT anchors #####
    #############################
    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])
    num_regions = 256
        
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - int(num_regions/2))
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = int(num_regions/2)
    
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
    
    
    #############################
    ###### Combine outputs ######
    #############################
    # y_rpn_overlap: objectiveness score - 0/1
    # y_is_box_valid: should box be included in loss - 0/1
    # y_rpn_regr: regression deltas - x,y,w,h
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=3), y_rpn_regr], axis=3)
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)
