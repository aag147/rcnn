# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:03:48 2018

@author: aag14
"""
import utils
import numpy as np
import copy as cp
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


#%%

def computeHOIAP(batch, GTMeta, nb_gt_samples, hoi_id, return_data=None):
    import filters_helper as helper
    
    props = np.array([x['score'] for x in batch])
    
    sorted_idxs = props.argsort()[::-1]
    nb_hois = len(sorted_idxs)
    
#    ious = [x/100 for x in range(50,100,5)]
#    nb_ious = len()
    
    tp = np.zeros((nb_hois))
    fp = np.zeros((nb_hois))
    
    evals = cp.copy(batch)
    
    
    for sort_idx, idx in enumerate(sorted_idxs):
        hoi = batch[idx]
        hbbox = hoi['hbbox']
        obbox = hoi['obbox']
        imageID = hoi['image_id']
    
        GTs = GTMeta[imageID]
        gt_hbboxes = GTs['gt_hbboxes']
        gt_obboxes = GTs['gt_obboxes']
        gt_rels = GTs['gt_rels']
        
        max_iou = 0.0
        best_idx = -1
        
        for gt_idx, gt_rel in enumerate(gt_rels):
            gt_hbbox = gt_hbboxes[gt_rel[0]:gt_rel[0]+1]
            gt_obbox = gt_obboxes[gt_rel[1]:gt_rel[1]+1]
            gt_label = gt_rel[2]
            
            if gt_label != hoi_id:
                continue
                        
            hum_overlap = helper._computeIoUs(hbbox, gt_hbbox)[0]
            obj_overlap = helper._computeIoUs(obbox, gt_obbox)[0]
            
            if min(hum_overlap, obj_overlap) > max_iou:
                max_iou = min(hum_overlap, obj_overlap)
                best_idx = gt_idx

        if return_data is not None:
            max_iou_bad = 0.0
            best_label_bad = -1
            
            for gt_idx, gt_rel in enumerate(gt_rels):
                gt_hbbox = gt_hbboxes[gt_rel[0]:gt_rel[0]+1]
                gt_obbox = gt_obboxes[gt_rel[1]:gt_rel[1]+1]
                gt_label = gt_rel[2]
                
                if gt_label == hoi_id:
                    continue
                            
                hum_overlap = helper._computeIoUs(hbbox, gt_hbbox)[0]
                obj_overlap = helper._computeIoUs(obbox, gt_obbox)[0]
                
                if min(hum_overlap, obj_overlap) > max_iou_bad:
                    max_iou_bad = min(hum_overlap, obj_overlap)
                    best_label_bad = gt_label            

        if max_iou >= 0.5:
            if gt_rels[best_idx,3] == 0:
                tp[sort_idx] = 1 # true positive
                GTMeta[imageID]['gt_rels'][best_idx,3] = 1
                evals[idx]['eval'] = hoi_id if return_data=='cfm' else 1 # true
            else:
                fp[sort_idx] = 1 # false positive (double)
                evals[idx]['eval'] = hoi_id if return_data=='cfm' else 1 # true (double)
        else:
            fp[sort_idx] = 1 # false positive
            
            if return_data is not None:
                if max_iou > 0.15 and return_data=='plt':
                    evals[idx]['eval'] = best_label_bad if return_data=='cfm' else -1 # low overlap
                elif max_iou_bad >= 0.5:
                    evals[idx]['eval'] = best_label_bad if return_data=='cfm' else -2 # misclass
                elif max_iou_bad > 0.15:
                    evals[idx]['eval'] = -1 if return_data=='cfm' else -3# misclass +  low overlap
                else:
                    evals[idx]['eval'] = -1 if return_data=='cfm' else -4# background
    
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    recall = tp / nb_gt_samples[hoi_id]
    precision = tp / (fp+tp)
    
#    print('recall', recall[:25])
#    print('pres', precision[:25])
#    print('tp', tp[:25])
#    print('fp', fp[:25])
#    print('tp/fp', (tp+fp)[:25])
#    print('nb_gt', nb_gt_samples[hoi_id])
    
    APs = np.zeros((11))
    for r in range(0,11):
        idxs = np.where(recall>= r/10.0)[0]
        if len(idxs) == 0:
            p = 0.0
        else:
            p = np.max(precision[idxs])
        APs[r] = p
        
    
    AP = np.mean(APs)
    
#    print(mAP, nb_gt_samples[hoi_id], APs, tp[-1], fp[-1])
    print('AP', AP)
    return AP, evals

    


def computeHOImAP(COCO_mapping, imagesMeta, class_mapping, hoi_mapping, cfg, return_data=False):
    import filters_helper as helper

    nb_gt_samples = np.zeros((len(hoi_mapping)))
    newGTMeta = {}
    for imageID, imageMeta in imagesMeta.items():
        gt_bboxes = imageMeta['objects']
        gt_rels = np.array(imageMeta['rels'])
        gt_hbboxes, gt_obboxes = helper._transformGTBBox(gt_bboxes, class_mapping, gt_rels, dosplit=True)
        gt_rels = helper._getRealRels(gt_rels)
        newGTMeta[imageID] = {'gt_hbboxes': gt_hbboxes, 'gt_obboxes': gt_obboxes, 'gt_rels': gt_rels}
        
        for gt_rel in gt_rels:
            nb_gt_samples[gt_rel[2]] += 1
    
    
    AP_map = np.zeros((len(hoi_mapping)))
    evalData = []
    for hoi_id, hoi_label in enumerate(hoi_mapping):

        batch = [x for x in COCO_mapping if x['category_id']==hoi_id]
        if len(batch)==0:
            continue
        AP, evals = computeHOIAP(batch, newGTMeta, nb_gt_samples, hoi_id, return_data)
        evalData.extend(evals)
        AP_map[hoi_id] = AP
        
    mAP = np.mean(AP_map)
    return mAP, AP_map, evalData



def computeRPNARHelperOld(batch, GTMeta, nb_gt_samples):
    import filters_helper as helper
    
    props = np.array([x['score'] for x in batch])
    
    sorted_idxs = props.argsort()[::-1]
    nb_rois = len(sorted_idxs)
    
    IoU = np.zeros((nb_rois))
    
    
    for i, idx in enumerate(sorted_idxs):
        
        if i % 1000 == 0:
            utils.update_progress_new(i, nb_rois, '')
        roi = batch[idx]
        bbox = roi['bbox']
        imageID = roi['image_id']
    
        GTs = GTMeta[imageID]
        gt_bboxes = GTs['gt_bboxes']
        
        overlap = helper._computeIoUs(bbox, gt_bboxes)

        max_iou = np.max(overlap)
        best_idx = np.argmax(overlap)
        
        if max_iou >= 0.5:
            if gt_bboxes[best_idx,4] == 0:
                IoU[idx] = max_iou # true positive
                GTMeta[imageID]['gt_bboxes'][best_idx,4] = 1
    
    R5 = np.sum(IoU>=0.5) / nb_gt_samples
    
    ious = [x/100 for x in range(50,100,5)]
    recalls = np.zeros((len(ious)))
    for idx, iou in enumerate(ious):
        r = np.sum(IoU>=iou) / nb_gt_samples
        recalls[idx] = r
        
    AR = np.mean(recalls)
    
    overlaps = [x - 0.5 for x in IoU if x > 0]
    AR = 2 * np.sum(overlaps) / nb_gt_samples
    
    return AR, R5, IoU

def computeRPNARHelper(predMeta, GTMeta):
    import filters_helper as helper
    
    nb_gt_samples = len(GTMeta)
    
    Ps = [100,300]
    
    IoU = np.zeros((2, nb_gt_samples))
    
    for idx in range(nb_gt_samples):
        
        if idx % 100 == 0:
            utils.update_progress_new(idx, nb_gt_samples, '')
        gt = GTMeta[idx]
        bbox = gt['bbox']
        imageID = gt['image_id']
    
        pred_bboxes = np.array(predMeta[imageID])
        
        overlaps = helper._computeIoUs(bbox, pred_bboxes)

        ol_idxs = np.argsort(overlaps)[::-1]
        max_iou = np.max(overlaps)

        done_Ps = [False for _ in range(len(Ps))]
        for ol_idx in ol_idxs:
            overlap   = overlaps[ol_idx]
            pred_bbox = pred_bboxes[ol_idx,:]
            top = pred_bbox[4]
            for P_idx, P in enumerate(Ps):
                if done_Ps[P_idx] or top > P:
                    continue
                if top <= P:
                    done_Ps[P_idx] = True
                if overlap >= 0.5:
                    IoU[P_idx,idx] = max_iou # true positive
                    
            if all(x for x in done_Ps):
                break
            
    
    R = [np.sum(x>=0.5) / nb_gt_samples for x in IoU]
    
#    ious = [x/100 for x in range(50,100,5)]
#    recalls = np.zeros((len(ious)))
#    for idx, iou in enumerate(ious):
#        r = np.sum(IoU>=iou) / nb_gt_samples
#        recalls[idx] = r
#        
#    AR = np.mean(recalls)
    
    overlaps = [[ol - 0.5 for ol in x if ol > 0] for x in IoU]
    AR = [2*np.sum(x) / nb_gt_samples for x in overlaps]
    
    return AR, R, IoU

def transformARGTs(gt_bboxes, class_mapping):
    
    new_gt_bboxes = []
    
    for b_idx, gt_bbox in enumerate(gt_bboxes):
        label = gt_bbox['label']
        label = class_mapping[label]
        xmin = gt_bbox['xmin']
        ymin = gt_bbox['ymin']
        xmax = gt_bbox['xmax']
        ymax = gt_bbox['ymax']
        
        coords = [xmin, ymin, xmax, ymax]
        coords = [round(float(x),2) for x in coords]
        coords += [label]
        
        new_gt_bboxes.append(coords)
    new_gt_bboxes = np.array(new_gt_bboxes)
    return new_gt_bboxes
        

def computeRPNAR(COCO_mapping, imagesMeta, class_mapping, cfg):    
    nb_gt_samples = 0
    newGTMeta = []
    newProposalMeta = {}
    
    for imageID, imageMeta in imagesMeta.items():
        gt_bboxes = imageMeta['objects']
        if len(gt_bboxes)==0:
            continue
        
        gt_bboxes = transformARGTs(gt_bboxes, class_mapping)
        
#        if imageID == '330818':
#            print(gt_bboxes)
        
        nb = gt_bboxes.shape[0]
        for gt_bbox in gt_bboxes:
            try:
                imageID = int(imageID)
            except ValueError:
                imageID = imageID
            newGTMeta.append({'image_id': imageID, 'bbox': gt_bbox})
        
        nb_gt_samples += nb
        
    
    for proposal in COCO_mapping:
        imageID = proposal['image_id']
        bbox = proposal['bbox']
        top = proposal['top']
        bbox = np.concatenate((bbox,[top]))
        
#        if int(imageID) == 330818:
#            print('pred', bbox)
        
        if imageID not in newProposalMeta:
            newProposalMeta[imageID] = []
        newProposalMeta[imageID].append(bbox)
        
    print('Number of GTs', nb_gt_samples)
    AR, R5, IoU = computeRPNARHelper(newProposalMeta, newGTMeta)
        
    return AR, R5, IoU   
            



class LogHistory():
    def __init__(self):
      self.train_loss = []
      self.train_acc = []
      self.val_loss = []
      self.val_acc = []
      
    def newEpoch(self, logs):
      self.train_loss.append(logs.get('loss'))
      self.train_acc.append(logs.get('acc'))
      self.val_loss.append(logs.get('val_loss'))
      self.val_acc.append(logs.get('val_acc'))

class EvalResults():
    def __init__(self, model, gen, yhat=None, y=None):
      self.tp = []
      self.fp = []
      self.fn = []
      self.precision = []
      self.recall = []
      
      self.mP = None
      self.mR = None
      self.F1 = None
      self.mAP = None
       
      self.nb_zeros = None
      
      self.model=model
      self.gen=gen
      self.Y_hat = None
      self.yhat = yhat
      self.y=y
      
      self._evaluate()

    def _evaluate(self):
      evalYHat = np.zeros([self.gen.nb_samples, self.gen.nb_classes])
      Y = np.zeros([self.gen.nb_samples, self.gen.nb_classes])
      iterGen = self.gen.begin()
      s_idx = 0
      
      if self.model is not None:
          for i in range(self.gen.nb_batches):
              utils.update_progress(i / self.gen.nb_batches)
              batch, y = next(iterGen)
              f_idx = s_idx + y.shape[-2]
              
              y_hat = self.model.predict_on_batch(x=batch)
              evalYHat[s_idx:f_idx, :] = y_hat
              Y[s_idx:f_idx, :] = y
              s_idx = f_idx

      else:
          evalYHat = self.yhat
          Y = self.y

      utils.update_progress(self.gen.nb_batches)
      print()
      accs, self.mP, self.mR, self.F1 = computeMultiLabelLoss(Y, evalYHat)
      self.mAP = computemAPLoss(Y, evalYHat)
      self.tp = accs[:,1]
      self.fp = accs[:,2]
      self.fn = accs[:,3]
      self.precision = accs[:,4]
      self.recall = accs[:,5]
      self.nb_zeros = np.count_nonzero(accs[:,1] == 0)
      self.Y_hat = evalYHat
      self.Y = Y


#%% COMPUTE ACCURACY
def computeLoss(Y, Y_hat, top):
    acc = 0.0
    for i in range(len(Y)):
        y = Y[i]
        p = Y_hat[i,:]
        I = np.argsort(p)[::-1]
        pTop = I[0:top]
        if y in pTop:
            acc = acc+1
    if len(Y)==0:
        acc = 2
    else:
        acc = acc / len(Y)
    loss = 1 - acc
    return acc
        
def computeConfusionMatrix(Y, Y_hat):
    Y = utils.getVectorLabels(Y)
    Y_hat = utils.getVectorLabels(Y_hat)
    return confusion_matrix(Y, Y_hat)

def computeMultiLabelLoss(Y, Y_hat):
    (nb_samples, nb_classes) = Y_hat.shape
    Y_hat = cp.copy(Y_hat)
    Y_hat[Y_hat>=0.5] = 1
    Y_hat[Y_hat<0.5] = 0
    accs = np.zeros((nb_classes,6))
    for x in range(nb_classes):
        y_total = 0
        tp = 0
        fp = 0
        fn = 0
        APs = []
        for i in range(nb_samples):
            y = Y[i,x]
            y_hat = Y_hat[i,x]
            if not y and y_hat:
                fp += 1
            if y and y_hat:
                tp += 1
            if y and not y_hat:
                fn += 1
            
        y_total = tp+fn
        p = tp / (tp+fp) if tp>0 else 0.0
        r = tp / (tp+fn) if tp>0 else 0.0
        
#        AP = average_precision_score(Y, Y_hat)
#        mAP = np.mean(AP)
        accs[x,:] = [y_total, tp, fp, fn, p, r]
        
    mP = np.mean(accs[:,4])
    mR = np.mean(accs[:,5])
    F1 = 0.0 if mP+mR==0 else 2 * ((mP * mR) / (mP + mR))
    return accs, mP, mR, F1


def computemAPLoss(Y, Y_hat):
    (nb_samples, nb_classes) = Y_hat.shape
    Y_hat = cp.copy(Y_hat)
    APs = np.zeros((nb_classes))
    for x in range(nb_classes):
        
        idxs = np.argsort(Y_hat[:,x])[::-1]
        
        nb_class_samples = len(np.where(Y[:,x]==1)[0])
        
        nb_preds = np.sum(Y_hat[:,x]>=0.0)
        
        tp = np.zeros(nb_preds)
        fp = np.zeros(nb_preds)
        
        for i in range(nb_preds):
            idx = idxs[i]
            y = Y[idx,x]

            if y==1:
                tp[i] = 1
            else:
                fp[i] = 1
            
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        recall = tp / nb_class_samples
        precision = tp / (fp+tp)
        
        Ps = np.zeros((11))
        for r in range(0,10):
            idxs = np.where(recall>= r/10.0)[0]
            if len(idxs) == 0:
                p = 0.0
            else:
                p = np.max(precision[idxs])
            Ps[r] = p
              
        AP = np.mean(Ps)
        APs[x] = AP
    mAP = np.mean(APs)
    return mAP
        


def computeConfusionMatrixLabels(Y, Y_hat):
    Y_hat = cp.copy(Y_hat)
    Y_hat[Y_hat>=0.5] = 1
    Y_hat[Y_hat<0.5] = 0
    
    print(Y_hat.shape)
    (smax,cmax) = Y_hat.shape
    colour_map = np.zeros([cmax+1,cmax+1])
    
    for sidx in range(smax):
        
        gt_classes = np.where(Y[sidx,:]==1)[0]
        pred_classes = np.where(Y_hat[sidx,:]==1)[0]
        if len(gt_classes) == 0:
            gt_classes = np.array([-1])
        for gt in gt_classes:
            if len(pred_classes) == 0:
                colour_map[gt+1,0] += 1
            for pred in pred_classes:
                if pred not in gt_classes or gt==pred:
                    colour_map[gt+1,pred+1] += 1
    colour_map.astype(np.int)
    return colour_map

def computeIndividualLabelLoss(Y, Y_hat):
    accs = np.zeros(16)
    for x in range(16):
        y_hat_x = []
        testY_x = []
        for i in range(len(Y)):
            if Y[i]==x:
                testY_x.append(x)
                y_hat_x.append(Y_hat[i,:])
        testY_x = np.array(testY_x)
        y_hat_x = np.array(y_hat_x)
        acc1  = computeLoss(testY_x, y_hat_x, 1)
        accs[x] = acc1
#        print("[%d] TOP 1: " %  (x), acc1)
    return accs

if __name__ == "__main__":
    print("hej")
