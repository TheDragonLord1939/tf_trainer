#!/usr/bin/env python
# coding=utf-8
# author: weianjun(weiaj@shareit.com)
# date: 2020.11.26

import os
import sys
sys.path.append(os.getcwd())

import json
import time
import signal
import datetime
import threading
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue
from kafka import KafkaConsumer, TopicPartition
from realtime.util.conf import Conf
from realtime.util.exception import *
from realtime.util.log import create_logger


#---------------------------------------------------------------------------------------------------#
class RealtimeReader(threading.Thread):
    kwargs_keys = [
        'app_name', 'kafka_hosts', 'kafka_ports', 'kafka_topic', 'logger', 'request_country', 'model_name', 'record_queue', 
        'starting_tstamp',
    ]

    def __init__(self, thread_name='', **kwargs):
        threading.Thread.__init__(self, name=thread_name)
        for item in RealtimeReader.kwargs_keys:
            if kwargs.get(item) is None: 
                raise KeyNotFound(item)
        hosts = kwargs['kafka_hosts'].split(',')
        ports = kwargs['kafka_ports'].split(',')
        group_id = '%s#%s#%s' % (kwargs['app_name'], kwargs['request_country'], kwargs['model_name'])
        brokers = ['%s:%s' % (h, p) for h, p in zip(hosts, ports)]
        self.__kafka_consumer = KafkaConsumer(
            kwargs['kafka_topic'],
            bootstrap_servers=brokers,
            group_id=group_id,
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_partition_fetch_bytes=2**18,
            retry_backoff_ms=100,
            request_timeout_ms=20000,
        )
        self.__fetch_cnt = 0
        self.__expir_cnt = 0
        self.__logger = kwargs['logger']
        self.__record_queue = kwargs['record_queue']
        self.__starting_tstamp = kwargs['starting_tstamp']
    
    def __fetch_records(self, timeout_ms=1000): 
        record_list = list()
        messages = self.__kafka_consumer.poll(timeout_ms=timeout_ms, max_records=100)
        if not messages:
            return record_list
        last_tstamp = 0
        for tp in messages:
            for item in messages[tp]:
                try:
                    record = json.loads(item.value.decode('utf8', 'ignore'))
                    request_time = int(int(record['request_time']) / 1000)
                    if int(request_time / 1e6) <= 0:
                        continue
                    if last_tstamp < request_time:
                        last_tstamp = request_time
                    if request_time < self.__starting_tstamp:
                        self.__expir_cnt += 1
                        continue
                except Exception as e:
                    self.__logger.info('Parse record=[%s,%s,%s,%s,%s] failed, e=%s' % (item.topic, item.partition, item.offset, item.key, record.get('request_time'), e))
                    continue
                record['request_time'] = request_time
                record_list.append(record)
        self.__fetch_cnt += len(record_list)
        if self.__expir_cnt >= 10000 and not self.__fetch_cnt:
            self.__logger.info('Fetched %d record(s), expired=%d, last_time=[%s]' % (
                self.__fetch_cnt, self.__expir_cnt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_tstamp))))
            self.__expir_cnt = 0
        if self.__fetch_cnt >= 10000:
            self.__logger.info('Fetched %d record(s), expired=%d, last_time=[%s]' % (
                self.__fetch_cnt, self.__expir_cnt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_tstamp))))
            self.__fetch_cnt = 0
            self.__expir_cnt = 0
        return record_list

    def run(self, ):
        while True:
            records = self.__fetch_records()
            if not records:
                continue
            for item in records:
                self.__record_queue.put(item)


#---------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    conf = Conf(os.path.join(os.getcwd(), 'realtime/conf/test.cfg'))
    conf.set('realtime_logger_file', os.path.join(conf.get('work_dir'), 'loader.log'))
    logger = create_logger(conf, 'realtime_logger')
    record_queue = Queue(3000)
    launch_time = '20201201000000'
    kwargs = {
        'app_name': conf.get('app_name'),
        'model_name': conf.get('model_name'),
        'kafka_hosts': conf.get('kafka_hosts'),
        'kafka_ports': conf.get('kafka_ports'),
        'kafka_topic': conf.get('kafka_topic'),
        'starting_tstamp': int(time.mktime(time.strptime(launch_time, '%Y%m%d%H%M%S'))),
        'request_country': conf.get('request_country'),
        'record_queue': record_queue,
        'logger': logger
    }
    threads = []
    for i in range(0, int(conf.get('loader_thread_num'))):
        threads.append(RealtimeReader(thread_name='RealtimeReader-%d' % (i), **kwargs))
    for t in threads:
        t.start()
    dump_cnt = 0
    debug_mode = True
    f = open(os.path.join(conf.get('work_dir'), 'record_queue.txt'), mode='w', encoding='utf8')
    print('#--------------- Starting reading real data ---------------#', file=sys.stderr)
    while True:
        record = record_queue.get()
        if dump_cnt < 10000:
            if debug_mode:
                del record['feature_input']
                request_time = record['request_time']
                request_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(request_time)) 
                record = json.dumps(record, ensure_ascii=False)
                f.write('%s\t' % (request_time) + record + '\n')
            else:
                record = json.dumps(record, ensure_ascii=False)
                f.write(record + '\n')
            dump_cnt += 1
        logger.info('Watchdog qsize: record=%d' % (record_queue.qsize()))
        if dump_cnt >= 10000:
            time.sleep(10)


#end
