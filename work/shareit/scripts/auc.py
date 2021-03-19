#!/usr/bin/env python
# coding=utf-8

import sys
import math
from collections import defaultdict
import numpy as np

def Auc(labels, preds, n_bins=100):
    positive_len = sum(labels)
    negative_len = len(labels) - positive_len
    total_case = positive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if nth_bin == n_bins:
            print('[WARNING] pred=%f, nth_bin=%d' % (preds[i], nth_bin))
            nth_bin = n_bins - 1
        if labels[i]>=1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]
    if total_case == 0:
        return 0.0
    return satisfied_pair / float(total_case)

def Avauc(labels, preds, idList):
    group_score = defaultdict(lambda:[])
    group_truth = defaultdict(lambda:[])
    for idx, truth in enumerate(labels):
        tempId = idList[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[tempId].append(score)
        group_truth[tempId].append(truth)
        group_flag = defaultdict(lambda: False) 
    for tempId in set(idList):
        truths = group_truth[tempId]
        label_sum = sum(truths)
        if label_sum > 0 and label_sum < len(truths):
            group_flag[tempId] = True
    impression_total = 0
    avg_truth_cnt = 0
    truth_cnt = 0
    total_auc = 0    
    for tempid in group_flag:
        if group_flag[tempid]:
            auc = Auc(np.asarray(group_truth[tempid]), np.asarray(group_score[tempid]), 1000)
            total_auc += auc * len(group_truth[tempid])
            impression_total += len(group_truth[tempid])
            avg_truth_cnt += len(group_truth[tempid])
            truth_cnt += 1
    group_auc = float(total_auc) / impression_total
    print('[NOTICE] Average group truth num : %.2f(%d/%d)' % (avg_truth_cnt * 1.0 / truth_cnt, avg_truth_cnt, truth_cnt))
    return group_auc
        
def Gauc(labels, preds, user_id_list):
    return  Avauc(labels, preds, user_id_list) 

def EvalRec(labels, preds):
    epsilon = 1e-7
    loss = 0.0
    sum_preds = 0.0
    pos_cnt, neg_cnt = 0, 0
    for i in range(0, len(labels)):
        pred = min(max(preds[i], epsilon), 1 - epsilon)
        if labels[i] == 1:
            pos_cnt += 1
            loss += -math.log(pred)
        elif labels[i] == 0:
            neg_cnt += 1
            loss += -math.log(1 - pred)
        sum_preds += pred
    loss /= (pos_cnt + neg_cnt)
    try:
        ctr = pos_cnt * 1.0 / (pos_cnt + neg_cnt)
    except Exception as e:
        print('[NOTICE] Calculate ctr failed, e=%s' % e)
        ctr = 0
    try:
        norm_loss = loss / -(ctr * math.log(ctr) + (1 - ctr) * math.log(1 - ctr))
    except Exception as e:
        norm_loss = 0
        print('[NOTICE] Calculate norm_loss failed, e=%s' % e)
    try:
        calibration = sum_preds * 1.0 / pos_cnt
    except Exception as e:
        calibration = 0
        print('[NOTICE] Calculate calibration failed, e=%s' % e)
    return ctr, loss, norm_loss, calibration
            
def main_PAL():
    cnt = 0
    labels = []
    preds  = []
    pctrs  = []
    groups = []
    col_map = {}
    for i, item in enumerate(sys.argv[2].split(',')):
        col_map[item] = i
    for line in sys.stdin:
        cnt += 1
        line = line.strip()
        tokens = line.split('\t')
        try:
            label   = tokens[col_map['label']]
            user_id = tokens[col_map['user_id']]
            pctr    = tokens[col_map['pctr']]
            bctr    = tokens[col_map['bctr']]
        except Exception as e:
            print('[NOTICE] Parser line="%s" failed, error=%s' % (line, e))
            sys.exit(0)
        if user_id == '0':
            continue
        preds.append(float(bctr))
        pctrs.append(float(pctr))
        labels.append(float(label))
        groups.append(user_id)
    print('[NOTICE] %d samples found' % cnt)
    print('[NOTICE] %d user_id found' % len(groups))
    print('[NOTICE] auc=%f' % (Auc(labels, pctrs, n_bins=1000)))
    print('[NOTICE] bias_auc=%f' % (Auc(labels, preds, n_bins=1000)))
    if len(groups) > 100:
        print('[NOTICE] gauc=%f' % (Gauc(labels, pctrs, groups)))
        print('[NOTICE] bias_gauc=%f' % (Gauc(labels, preds, groups)))

def main_user_type(valid_flag):
    user_type_dict = {'1145810468633': 'client', '1146415534720': 'tool', '1147020600807': 'content'}
    counts = {}
    labels = {}
    groups = {}
    preds  = {}
    col_map = {}
    for i, item in enumerate(sys.argv[2].split(',')):
        col_map[item] = i
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split('\t')
        try:
            label     = tokens[col_map['label']]
            user_id   = tokens[col_map['user_id']]
            user_type = tokens[col_map['user_type']]
            pctr      = tokens[col_map['pctr']]
            user_type = user_type_dict.get(user_type, None)
        except Exception as e:
            print('[NOTICE] Parser line="%s" failed, error=%s' % (line, e))
            sys.exit(0)
        if user_type is None:
            continue
        if user_id == '0':
            continue
        for item in ['all', user_type]:
            counts.setdefault(item, 0)
            labels.setdefault(item, [])
            groups.setdefault(item, [])
            preds.setdefault(item, [])
            labels[item].append(float(label))
            preds[item].append(float(pctr))
            groups[item].append(user_id)
            counts[item] += 1
    results = []
    for item in counts:
        print('[NOTICE] %d samples of %s found' % (counts[item], item))
        print('[NOTICE] %d user_id of %s found' % (len(groups[item]), item))
        if item == 'all': 
            key = ''
        else:
            key = '#' + item
        ctr, loss, norm_loss, calibration = EvalRec(labels[item], preds[item])
        auc = Auc(labels[item], preds[item], n_bins=1000)
        if len(groups[item]) > 100:
            gauc = Gauc(labels[item], preds[item], groups[item])
        else:
            gauc = 0
        results.append('ctr%s=%f' % (key, ctr))
        results.append('loss%s=%.10f' % (key, loss))
        results.append('norm_loss%s=%.10f' % (key, norm_loss))
        results.append('calibration%s=%f' % (key, calibration))
        results.append('auc%s=%f'  % (key,  auc))
        results.append('gauc%s=%f' % (key, gauc))
    print('[NOTICE] %s %s' % (valid_flag, ' '.join(results)))

def main(valid_flag):
    cnt = 0
    labels = []
    preds  = []
    groups = []
    col_map = {}
    for i, item in enumerate(sys.argv[2].split(',')):
        col_map[item] = i
    for line in sys.stdin:
        cnt += 1
        line = line.strip()
        tokens = line.split('\t')
        try:
            label   = tokens[col_map['label']]
            user_id = tokens[col_map['user_id']]
            pctr    = tokens[col_map['pctr']]
        except Exception as e:
            print('[NOTICE] Parser line="%s" failed, error=%s' % (line, e))
            sys.exit(0)
        if user_id == '0':
            continue
        labels.append(float(label))
        preds.append(float(pctr))
        groups.append(user_id)
    print('[NOTICE] %d samples found' % cnt)
    print('[NOTICE] %d user_id found' % len(groups))
    auc = Auc(labels, preds, n_bins=1000)
    if len(groups) > 100:
        gauc = Gauc(labels, preds, groups)
    else:
        gauc = 0
    ctr, loss, norm_loss, calibration = EvalRec(labels, preds)
    print('[NOTICE] %s positive_rate=%f loss=%.10f norm_loss=%.10f calibration=%f auc=%f gauc=%f' % (
        valid_flag, ctr, loss, norm_loss, calibration, auc, gauc
    ))

if len(sys.argv) < 4:
    print('[NOTICE] Usage: %s <auc_type> <parser_format> <valid_flag>' % sys.argv[0])
    sys.exit(0)

if sys.argv[1] == 'main':
    main(sys.argv[3])
if sys.argv[1] == 'main_PAL':
    main_PAL()
if sys.argv[1] == 'main_user_type':
    main_user_type(sys.argv[3])
