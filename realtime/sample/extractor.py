#!/usr/bin/env python
# coding=utf-8
# author: weianjun(weiaj@shareit.com)
# date: 2020.11.27

import os
import sys
sys.path.append(os.getcwd())

import time
import json
import random
import signal
import string
import datetime
import threading
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue
from realtime.util.conf import Conf
from realtime.util.exception import *
from realtime.util.log import create_logger
from pyfeature_extract import PyFeatureExtract

#---------------------------------------------------------------------------------------------------#
class FeatureExtractor(threading.Thread):
    kwargs_keys = [
        'logger', 'request_country', 'request_scenes', 'record_queue', 'instance_queue', 'label_key', 
        'extract_feature_config', 'pos_rule', 'neg_rule',
    ]

    def __init__(self, thread_name='', **kwargs):
        threading.Thread.__init__(self, name=thread_name)
        for item in FeatureExtractor.kwargs_keys:
            if kwargs.get(item) is None:
                raise KeyNotFound(item)
        try:
            with open(kwargs['extract_feature_config']) as f:
                self.__extractor = PyFeatureExtract(json.dumps(json.load(f), ensure_ascii=False))
        except Exception as e:
            raise RuntimeError('Init feature extractor failed, error=%s' % e)
        self.__logger = kwargs['logger']
        self.__request_country = kwargs['request_country']
        self.__scenes = set(kwargs['request_scenes'].split(','))
        self.__record_queue = kwargs['record_queue']
        self.__instance_queue = kwargs['instance_queue']
        self.__label_key = kwargs['label_key']
        self.__pos_rule = kwargs['pos_rule']
        self.__neg_rule = kwargs['neg_rule']
        self.__pos_rate = kwargs.get('pos_sample_rate', 1.0)
        self.__neg_rate = kwargs.get('neg_sample_rate', 1.0)
        self.__is_debug = kwargs.get('debug_mode', False)
         
    def __filter_record(self, record):
        '''
            return: False - throw the record, True - retain the record
        '''
        if record.get('feature_input') is None or record.get('labels') is None:
            if self.__is_debug:
                self.__logger.info('Throw record because feature_input or labels is None')
            return False
        if record.get('request_country') is None or record['request_country'] != self.__request_country:
            if self.__is_debug:
                self.__logger.info('Throw record because request_country=%s' % (record.get('request_country')))
            return False
        if record.get('reco_scene') is None or record['reco_scene'] not in self.__scenes:
            if self.__is_debug:
                self.__logger.info('Throw record because reco_scene=%s' % (record.get('reco_scene')))
            return False
        return True

    def __sample_record(self, record):
        '''
            return: None - throw the record, not None - label value
        '''
        label_val = None
        labels = record['labels']
        rand_strs = ''.join([random.choice(string.ascii_letters) for i in range(3)])
        if labels.get(self.__label_key) is None:
            label_def = '%s_%s = 0'  % (rand_strs, self.__label_key)
            label_val = 0
        else:
            label_def = '%s_%s = %s' % (rand_strs, self.__label_key, labels[self.__label_key])
            label_val = labels[self.__label_key]
        try:
            exec(label_def)
        except Exception as e:
            self.__logger.warn('Define label failed, cmd=[%s], error=%s' % (label_def, e))
            return None
        try:
            is_pos = eval('%s_%s' % (rand_strs, self.__pos_rule))
            is_neg = eval('%s_%s' % (rand_strs, self.__neg_rule))
        except Exception as e:
            self.__logger.warn('Calculate label failed, rules=[%s/%s]' % (pos_rule, neg_rule))
            return None
        if isinstance(label_val, str):
            try:
                label_val = int(label_val)
            except Exception as e:
                self.__logger.warn('label value is illegal, value="%s", key=%s' % (label_val, self.__label_key))
                return None
        if is_pos:
            return label_val if self.__pos_rate >= random.random() else None
        if is_neg:
            return label_val if self.__neg_rate >= random.random() else None
        return label_val

    def run(self, ):
        while True:
            record = self.__record_queue.get()
            if not self.__filter_record(record):
                continue
            label = self.__sample_record(record)
            if label is None:
                continue
            hash_feats = self.__extractor.extract(record['feature_input'])
            instance = json.loads(hash_feats)
            instance['labels'] = {self.__label_key: label}
            instance['request_time'] = record['request_time']
            self.__instance_queue.put(instance)


#---------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    conf = Conf(os.path.join(os.getcwd(), 'realtime/conf/test.cfg'))
    conf.set('instance_logger_file', os.path.join(conf.get('work_dir'), 'extractor.log'))
    logger = create_logger(conf, 'instance_logger')
    record_queue = Queue(20000)
    instance_queue = Queue(20000)
    kwargs = {
        'extract_feature_config': conf.get('extract_feature_config'),
        'pos_sample_rate': float(conf.get('pos_sample_rate')),
        'neg_sample_rate': float(conf.get('neg_sample_rate')),
        'request_scenes': conf.get('request_scenes'),
        'request_country': conf.get('request_country'),
        'label_key': conf.get('label_key'),
        'pos_rule': conf.get('pos_rule'),
        'neg_rule': conf.get('neg_rule'),
        'record_queue': record_queue,
        'instance_queue': instance_queue,
        'debug_mode': True,
        'logger': logger,
    }
    dump_cnt = 0
    record_file = os.path.join(conf.get('work_dir'), 'record_queue.txt')
    if os.path.exists(record_file):
        for line in open(record_file, mode='r', encoding='utf8'):
            record_queue.put(json.loads(line.strip()))
            if dump_cnt % 1000 == 0:
                print('%d lines done' % (dump_cnt))
            dump_cnt += 1
    print('%d records found' % record_queue.qsize())
    threads = []
    for i in range(0, int(conf.get('extractor_thread_num'))):
        threads.append(FeatureExtractor(thread_name='FeatureExtractor-%d' % (i), **kwargs))
    for t in threads:
        t.start()
    f = open(os.path.join(conf.get('work_dir'), 'instance_queue.txt'), mode='w', encoding='utf8')
    dump_cnt = 0
    while True:
        instance = instance_queue.get()
        if dump_cnt < 5000:
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')
            dump_cnt += 1
        logger.info('Watchdog qsize: record=%d, instance=%d' % (
            record_queue.qsize(), instance_queue.qsize()))
        if dump_cnt >= 5000:
            time.sleep(10)

#end
