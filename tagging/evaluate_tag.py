# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-05-26
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import numpy as np


msrvtt_gt = 'msrvtt_tag_gt_4_msrvtt.npy'
msrvtt_tag_feats = ['msrvtt_e100_tag_feats.npy', 'msrvtt_e200_tag_feats.npy', 
                    'msrvtt_e400_tag_feats.npy', 'msrvtt_e800_tag_feats.npy', 
                    'msrvtt_e1000_tag_feats.npy'] 

msvd_gt = 'msvd_tag_gt_4_msvd.npy'
msvd_tag_feats = ['msvd_semantic_tag_e100.npy', 'msvd_semantic_tag_e200.npy', 
                'msvd_semantic_tag_e400.npy', 'msvd_semantic_tag_e800.npy', 
                'msvd_semantic_tag_e1000.npy']

msrvtt_res = {}
label = np.load(msrvtt_gt)
for mfeat in msrvtt_tag_feats:
    pred = np.load(mfeat)
    ap_list = []
    for j in range(pred.shape[1]):
        precision, recall, thresholds = precision_recall_curve(label[:,j], pred[:,j])
        ap = np.mean(precision)
        ap_list.append(ap)
    count = int(mfeat.split('_')[1][1:])
    msrvtt_res[count] = np.mean(ap_list)

msvd_res = {}
label = np.load(msvd_gt)
for mfeat in msvd_tag_feats:
    pred = np.load(mfeat)
    ap_list = []
    for j in range(pred.shape[1]):
        precision, recall, thresholds = precision_recall_curve(label[:,j], pred[:,j])
        ap = np.mean(precision)
        ap_list.append(ap)
    count = int(mfeat.split('_')[3].split('.')[0][1:])
    msvd_res[count] = np.mean(ap_list)


from pprint import pprint
print('MSRVTT results:')
pprint(msrvtt_res)

print('MSVD results:')
pprint(msvd_res)
