# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
generate config for training dataset
"""
import os
import sys

data_dir = sys.argv[1]
work_dir = sys.argv[2]

train_path = [os.path.join(data_dir, 'train_data')]
valid_path = [os.path.join(data_dir, 'pred_data'), os.path.join(data_dir, 'pre_data')]

print('train_path=%s' % (train_path), file=sys.stderr)
print('valid_path=%s' % (valid_path), file=sys.stderr)

cfg_dir = os.path.join(work_dir, 'cfg')
if not os.path.exists(cfg_dir):
    os.mkdir(cfg_dir)

# Get config file for train dataset
train_cfg = open(os.path.join(cfg_dir, 'train'), 'w')
for path in train_path:
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if 'part' in file:
                fullpath = os.path.join(dirpath, file)
                train_cfg.write(fullpath+'\n')
train_cfg.close()

# Get config file for valid dataset
valid_cfg = open(os.path.join(cfg_dir, 'valid'), 'w')
for path in valid_path:
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if 'part' in file:
                fullpath = os.path.join(dirpath, file)
                valid_cfg.write(fullpath+'\n')
valid_cfg.close()
