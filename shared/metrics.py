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

def computeHOIAP(batch, GTMeta, nb_gt_samples, hoi_id):
    import filters_helper as helper
    
    props = np.array([x['score'] for x in batch])
    
    sorted_idxs = props.argsort()[::-1]
    nb_hois = len(sorted_idxs)
    
    tp = np.zeros((nb_hois))
    fp = np.zeros((nb_hois))
    
    
    for idx in sorted_idxs:
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
        
        if max_iou > 0.5:
            if gt_rels[best_idx,3] == 0:
                tp[idx] = 1 # true positive
                GTMeta[imageID]['gt_rels'][best_idx,3] = 1
            else:
                fp[idx] = 1 # false positive (double)
        else:
            fp[idx] = 1 # false positive
    
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    recall = tp / nb_gt_samples[hoi_id]
#    recall = tp / 1
    precision = tp / (fp+tp)
    
    APs = np.zeros((11))
    for r in range(0,11):
        idxs = np.where(recall>= r/10.0)[0]
        if len(idxs) == 0:
            p = 0.0
        else:
            p = np.max(precision[idxs])
        APs[r] = p
        
    
    mAP = np.mean(APs)
    return mAP

    


def computeHOImAP(COCO_mapping, imagesMeta, class_mapping, hoi_mapping, cfg):
    import filters_helper as helper

    nb_gt_samples = np.zeros((len(hoi_mapping)))
    newGTMeta = {}
    for imageID, imageMeta in imagesMeta.items():
        gt_bboxes = imageMeta['objects']
        gt_rels = imageMeta['rels']
        gt_hbboxes, gt_obboxes = helper._transformGTBBox(gt_bboxes, class_mapping, gt_rels, dosplit=True)
        gt_rels = helper._getRealRels(gt_rels)
        newGTMeta[imageID] = {'gt_hbboxes': gt_hbboxes, 'gt_obboxes': gt_obboxes, 'gt_rels': gt_rels}
        
        for gt_rel in gt_rels:
            nb_gt_samples[gt_rel[2]] += 1
    
    AP_map = np.zeros((len(hoi_mapping)))
        
    for hoi_id, hoi_label in enumerate(hoi_mapping):

        batch = [x for x in COCO_mapping if x['category_id']==hoi_id]
        if len(batch)==0:
            continue
        AP = computeHOIAP(batch, newGTMeta, nb_gt_samples, hoi_id)
        AP_map[hoi_id] = AP
        
    mAP = np.mean(AP_map)
    return mAP, AP_map
            



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
    def __init__(self, model, gen):
      self.tp = []
      self.fp = []
      self.fn = []
      self.precision = []
      self.recall = []
      
      self.mP = None
      self.mR = None
      self.F1 = None
       
      self.nb_zeros = None
      
      self.model=model
      self.gen=gen
      self.Y_hat = None
      
      self._evaluate()

    def _evaluate(self):
      evalYHat = np.zeros([self.gen.nb_samples, self.gen.nb_classes])
      Y = np.zeros([self.gen.nb_samples, self.gen.nb_classes])
      iterGen = self.gen.begin()
      s_idx = 0
      for i in range(self.gen.nb_batches):
          utils.update_progress(i / self.gen.nb_batches)
          batch, y = next(iterGen)
          y_hat = self.model.predict_on_batch(x=batch)
          f_idx = s_idx + y.shape[-2]
#          print(y_hat.shape)
          evalYHat[s_idx:f_idx, :] = y_hat
          Y[s_idx:f_idx, :] = y
          s_idx = f_idx
      utils.update_progress(self.gen.nb_batches)
      print()
      accs, self.mP, self.mR, self.F1 = computeMultiLabelLoss(Y, evalYHat)
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
