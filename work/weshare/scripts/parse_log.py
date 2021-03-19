#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 SHAREit <fuhl@>
#

"""
extract experimental results from log automatically
# Usage:
# $ python parse_log.py some_directory_a/train.log some_directory_b/test.log
# or (the 2nd parameter is optional)
# $ python parse_log.py some_directory_a/train.log
"""

import os
import re
import sys
import pandas as pd
from datetime import datetime

COLUMNS = ['model', 'train loss', 'valid loss', 'test loss',
           'train norm loss', 'valid norm loss', 'test norm loss',
           'train calibration', 'valid calibration', 'test calibration',
           'train base logloss', 'valid base logloss', 'test base logloss',
           'train ctr', 'valid ctr', 'test ctr',
           'train roc', 'valid roc', 'test roc',
           'train pr', 'valid pr', 'test pr',
           'rate', 'stop step', 'cost time']


def parse_log(model_name, train_log, test_log=None):
    df = pd.DataFrame(columns=COLUMNS)

    with open(train_log, 'r') as f:
        find_InitTime = False
        train_base_logloss = ''
        for line in f.readlines():
            # ********************************** cost time ************************************************
            time = re.findall(r'\d{4}\d{2}\d{2}\s\d{2}:\d{2}:\d{2}', line)
            if len(time) > 0:
                if not find_InitTime:
                    start_time = datetime.strptime(time[0], '%Y%m%d %H:%M:%S')
                    find_InitTime = True
                end_time = time[0]

            # ********************************** base logloss & ctr ***************************************
            if 'logloss' in line:
                if train_base_logloss == '':
                    train_base_logloss = re.findall(r"\d+\.?\d*", line)[0]
                    train_ctr = re.findall(r"\d+\.?\d*", line)[1]
                else:
                    valid_base_logloss = re.findall(r"\d+\.?\d*", line)[0]
                    valid_ctr = re.findall(r"\d+\.?\d*", line)[1]

            # ************************************* others ************************************************
            if 'Train' in line:
                tmp_train_loss = re.findall(r'(?<=loss=)\d+\.?\d*', line)[0]
                tmp_train_norm_loss = re.findall(
                    r'(?<=norm_loss=)\d+\.?\d*', line)[0]
                tmp_train_roc = re.findall(r'(?<=roc=)\d+\.?\d*', line)[0]
                tmp_train_pr = re.findall(r'(?<=pr=)\d+\.?\d*', line)[0]
                tmp_train_calibration = re.findall(
                    r'(?<=calibration=)\d+\.?\d*', line)[0]
                rate = round(
                    float(re.findall(r'(?<=rate=)\d+\.?\d*', line)[0]), 2)
            if 'better loss' in line:
                train_loss = tmp_train_loss
                train_norm_loss = tmp_train_norm_loss
                train_roc = tmp_train_roc
                train_pr = tmp_train_pr
                train_calibration = tmp_train_calibration
                stop_step = re.findall(r'(?<=steps=)\d+\.?\d*', line)[0]
            if 'best valid' in line:
                valid_loss = re.findall(r'(?<=loss=)\d+\.?\d*', line)[0]
                valid_norm_loss = re.findall(
                    r'(?<=norm_loss=)\d+\.?\d*', line)[0]
                valid_roc = re.findall(r'(?<=roc=)\d+\.?\d*', line)[0]
                valid_pr = re.findall(r'(?<=pr=)\d+\.?\d*', line)[0]
                valid_calibration = re.findall(
                    r'(?<=calibration=)\d+\.?\d*', line)[0]

    cost_time = str(round((datetime.strptime(
        end_time, '%Y%m%d %H:%M:%S') - start_time).total_seconds() / 3600, 2)) + 'h'
    df = df.append(pd.DataFrame({'model': [model_name],
                                 'train loss': [train_loss], 'valid loss': [valid_loss],
                                 'train norm loss': [train_norm_loss], 'valid norm loss': [valid_norm_loss],
                                 'train calibration': [train_calibration], 'valid calibration': [valid_calibration],
                                 'train base logloss': [train_base_logloss], 'valid base logloss': [valid_base_logloss],
                                 'train ctr': [train_ctr], 'valid ctr': [valid_ctr],
                                 'train roc': [train_roc], 'valid roc': [valid_roc],
                                 'train pr': [train_pr], 'valid pr': [valid_pr],
                                 'rate': [rate], 'stop step': [stop_step], 'cost time': [cost_time]}),
                   ignore_index=True, sort=False)

    if test_log is None:
        cols = [c for c in df.columns if c.lower()[:4] != 'test']
        df = df[cols]
        return df
    else:
        with open(test_log, 'r') as f:
            for line in f.readlines():
                if 'logloss' in line:
                    test_base_logloss = re.findall(r"\d+\.?\d*", line)[0]
                    test_ctr = re.findall(r"\d+\.?\d*", line)[1]
                if 'Eval' in line:
                    test_loss = re.findall(r'(?<=loss=)\d+\.?\d*', line)[0]
                    test_norm_loss = re.findall(
                        r'(?<=norm_loss=)\d+\.?\d*', line)[0]
                    test_roc = re.findall(r'(?<=roc=)\d+\.?\d*', line)[0]
                    test_pr = re.findall(r'(?<=pr=)\d+\.?\d*', line)[0]
                    test_calibration = re.findall(
                        r'(?<=calibration=)\d+\.?\d*', line)[0]
        df.loc[df['model'] == model_name, 'test loss'] = test_loss
        df.loc[df['model'] == model_name, 'test norm loss'] = test_norm_loss
        df.loc[df['model'] == model_name,
               'test calibration'] = test_calibration
        df.loc[df['model'] == model_name,
               'test base logloss'] = test_base_logloss
        df.loc[df['model'] == model_name, 'test ctr'] = test_ctr
        df.loc[df['model'] == model_name, 'test roc'] = test_roc
        df.loc[df['model'] == model_name, 'test pr'] = test_pr
        return df


if __name__ == '__main__':
    train_log = sys.argv[1]  # 1st parameter: train log file
    model_name = os.path.splitext(os.path.basename(train_log))[0]
    if len(sys.argv) > 2:
        test_log = sys.argv[2]  # 2nd parameter(optional): test log file
        res_df = parse_log(model_name, train_log, test_log)
    else:
        res_df = parse_log(model_name, train_log)
    target_log_path = os.path.abspath(train_log)
    print(res_df.head().transpose())
    filename, file_extension = os.path.splitext(target_log_path)
    res_df.to_csv(filename + '.csv', encoding='utf-8', index=False)
    print('Save to ' + filename + '.csv')

