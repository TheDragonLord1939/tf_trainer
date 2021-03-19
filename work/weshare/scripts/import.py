#!/usr/bin/env python
# coding=utf-8

import os
import sys
import time
import datetime
from socket import socket

if len(sys.argv) != 3:
    print('Usage: %s <base_key> <date>' % (sys.argv[0]))
    sys.exit(0)

indicator_dict = {
    'pr': 'pr',
    'auc': 'auc',
    'gauc': 'gauc',
    'loss': 'loss',
    'norm_loss': 'norm_loss',
    'model_size': 'model_size',
    'calibration': 'calibration',
    'feature_num': 'feature_num',
    'train_samples': 'train_samples',
    'auc#content': 'auc#content',
    'gauc#content': 'gauc#content',
    'duration/pos/label': 'label',
    'duration/pos/score': 'score',
    'duration/pos/rmse': 'rmse',
    'duration/loss': 'loss',
    'positive_rate': 'positive_rate',
}

base_key = sys.argv[1]
if len(sys.argv[2]) == 8:
    d = datetime.datetime.strptime(sys.argv[2], '%Y%m%d')
elif len(sys.argv[2]) == 15:
    d = datetime.datetime.strptime(sys.argv[2], '%Y%m%d_%H%M%S')
else:
    raise RuntimeError('date format error, must be 20201214 or 20201214_081030')
tstamp = time.mktime(d.timetuple())

data_list = []
for line in sys.stdin:
    line = line.strip()
    if line.find('[monitor]') == -1:
        continue
    tokens = line.split(' ')
    if len(tokens) != 2:
        continue
    try:
        name, val = tokens[1].split('=')
    except Exception as e:
        print('[WARNING] %s is illegal' % tokens[3])
        continue
    if name not in indicator_dict:
        print('[WARNING] %s is illegal' % name)
        continue
    name = indicator_dict[name]
    if name == 'model_size':
        val = int(val) * 1.0 / 1024 / 1024
    data_list.append('%s.%s %s %d' % (base_key, name, val, tstamp))
print('[NOTICE] %d monitor indicators found' % (len(data_list)))

sock = socket()
sock.connect(('10.21.97.9', 2003))

message = "\n".join(data_list) + "\n"
print(message)
sock.sendall(message.encode())
