#!/usr/bin/env python
# coding=utf-8
# author: weianjun(weiaj@ushareit.com)
# date: 2020.11.30

import os
import sys
sys.path.append(os.getcwd())

import time
import json
import signal
import threading
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue
import tensorflow as tf
from realtime.util.conf import Conf
from realtime.util.exception import *
from realtime.util.log import create_logger

#---------------------------------------------------------------------------------------------------#
class TfrecordMaker(threading.Thread):
    kwargs_keys = [
        'instance_queue', 'tfrecord_queue', 'feature_schema_file', 'prebatch', 'logger'
    ]

    def __init__(self, thread_name='', **kwargs):
        threading.Thread.__init__(self, name=thread_name)
        for item in TfrecordMaker.kwargs_keys:
            if kwargs.get(item) is None:
                raise KeyNotFound(item)
        schema_file = kwargs['feature_schema_file']
        if not os.path.exists(schema_file):
            raise FileNotFound(schema_file)
        self.__logger = kwargs['logger']
        self.__load_feature_schema(schema_file)
        self.__prebatch = kwargs['prebatch']
        self.__instance_queue = kwargs['instance_queue']
        self.__tfrecord_queue = kwargs['tfrecord_queue']
        self.__is_debug = kwargs.get('debug_mode', False)

    def __load_feature_schema(self, schema_file):
        self.__feat_schema = dict()
        for line in open(schema_file, mode='r', encoding='utf8'):
            line = line.strip()
            try:
                obj = json.loads(line)
                if obj['hash_threshold'] < 0:
                    obj['hash_threshold'] = 0
                if obj['sequence_length'] < 1:
                    obj['sequence_length'] = 1
                self.__feat_schema[obj['feature_name']] = [
                    obj['feature_type'], obj['sequence_length'], obj['hash_threshold']]
            except Exception as e:
                self.__logger.info('Parse line=%s failed' % (line))
                continue
        self.__logger.info('Load feature schema OK, %d features found' % (len(self.__feat_schema)))
    
    def __init_feature_index(self, index_dict):
        index_dict.clear()
        for fname in self.__feat_schema:
            ftype, flen, fsize = self.__feat_schema[fname]
            if ftype == 'int' or ftype == 'double':
                index_dict[fname] = [0] * self.__prebatch
            elif ftype == 'sequence':
                index_dict[fname] = [0] * self.__prebatch * flen

    def __adapt_from_hash(self, ins_cnt, instance, index_dict):
        succ_cnt = 0
        if ins_cnt >= self.__prebatch:
            self.__loger.warn('instance_count=%d is more than prebatch size' % ins_cnt)
            return succ_cnt
        if not isinstance(instance['features'], list) or not instance['features']:
            self.__logger.info('instance.features is illegal')
            return succ_cnt
        features = instance['features'][0]
        if not isinstance(features, dict):
            self.__logger.info('instance.features is illegal')
            return succ_cnt
        labels = instance['labels']
        dense_features = features.get('dense_features', {})
        sparse_features = features.get('sparse_features', {})
        sequence_features = features.get('sequence_features', {})
        for item in labels:
            if item not in self.__feat_schema:
                continue
            ftype, flen, fsize = self.__feat_schema[item]
            index_list = index_dict[item]
            label = labels[item]
            try:
                label = float(label) if ftype == 'double' else int(label)
            except Exception as e:
                self.__logger.info('[%s] label=%s is not float or int type' % (self.getName(), label))
                continue
            index_list[ins_cnt] = label
            succ_cnt += 1
        if not succ_cnt:
            self.__logger.info('instance.label is NULL, labels=%s' % (labels))
            return succ_cnt
        for fname in self.__feat_schema:
            if fname in labels:
                continue
            ftype, flen, fsize = self.__feat_schema[fname]
            index_list = index_dict[fname]
            if ftype == 'int' and fname in sparse_features:
                if fsize > 0:
                    fid = sparse_features[fname] % (fsize - 1) + 1
                    index_list[ins_cnt] = fid
                else:
                    index_list[ins_cnt] = sparse_features[fname]
                succ_cnt += 1
            elif ftype == 'double' and fname in dense_features:
                index_list[ins_cnt] = dense_features[fname]
                succ_cnt += 1
            elif ftype == 'sequence' and fname in sequence_features:
                value = sequence_features[fname]
                start_id = ins_cnt * flen
                for i in range(0, len(value)):
                    if i < flen:
                        fid = value[i] % (fsize - 1) + 1
                        index_list[start_id + i] = fid
                succ_cnt += 1
            elif self.__is_debug:
                self.__logger.info('Miss feature=%s, info=%s in feature_input' % (fname, self.__feat_schema[fname]))
                continue
        return succ_cnt
    
    def run(self, ):
        instance_cnt = 0
        index_dict = dict()
        min_tstamp = sys.maxsize
        self.__init_feature_index(index_dict)
        while True:
            if instance_cnt >= self.__prebatch:
                feature_map = dict()
                for fname in self.__feat_schema:
                    ftype, flen, fsize = self.__feat_schema[fname]
                    if ftype == 'int':
                        feature_map[fname] = tf.train.Feature(int64_list=tf.train.Int64List(value=index_dict[fname]))
                    elif ftype == 'sequence':
                        feature_map[fname] = tf.train.Feature(int64_list=tf.train.Int64List(value=index_dict[fname]))
                    elif ftype == 'double':
                        feature_map[fname] = tf.train.Feature(float_list=tf.train.FloatList(value=index_dict[fname]))
                    else:
                        continue
                example = tf.train.Example(features=tf.train.Features(feature=feature_map))
                self.__tfrecord_queue.put((min_tstamp, example.SerializeToString()))
                self.__init_feature_index(index_dict)
                min_tstamp = sys.maxsize
                instance_cnt = 0
            instance = self.__instance_queue.get()
            tstamp = instance['request_time']
            if tstamp < min_tstamp:
                min_tstamp = tstamp
            ret = self.__adapt_from_hash(instance_cnt, instance, index_dict)
            if ret < 2:
                self.__logger.info('Adapt instance failed, ret=%d/%d' % (ret, len(self.__feat_schema)))
            else:
                instance_cnt += 1


#---------------------------------------------------------------------------------------------------#
class TfrecordSaver(threading.Thread):
    kwargs_keys = [
        'tfrecord_queue', 'root_save_dir', 'logger', 'save_interval_min', 'starting_tstamp'
    ]

    def __init__(self, thread_name='', **kwargs):
        threading.Thread.__init__(self, name=thread_name)
        for item in TfrecordSaver.kwargs_keys:
            if kwargs.get(item) is None:
                raise KeyNotFound(item)
        self.__logger = kwargs['logger']
        self.__root_save_dir = kwargs['root_save_dir']
        self.__launch_tstamp = kwargs['starting_tstamp']
        self.__launch_time = time.strftime("%Y%m%d%H%M%S", time.localtime(self.__launch_tstamp))
        self.__epoch_interval = kwargs['save_interval_min'] * 60
        self.__tfrecord_queue = kwargs['tfrecord_queue']
        self.__tfrecord_cnt = 0
        self.__tolerance = 3
        self.__error_try = 0
        self.__writer = None
        self.__create_writer()
    
    def __update_status(self):
        self.__launch_tstamp += self.__epoch_interval
        self.__launch_time = time.strftime("%Y%m%d%H%M%S", time.localtime(self.__launch_tstamp))
        self.__tfrecord_cnt = 0
        self.__error_try = 0

    def __create_writer(self):
        if self.__writer is not None:
            self.__writer.close()
        self.__writer = tf.io.TFRecordWriter(os.path.join(self.__save_dir(), self.getName()), options='GZIP')
        self.__logger.info('Created a new TFRecordWriter for tstamp=%d, time=%s' % (self.__launch_tstamp, self.__launch_time))

    def __write_success(self):
        os.system('touch %s' % os.path.join(self.__save_dir(), '_SUCCESS.%s' % self.getName()))
    
    def __save_dir(self):
        save_dir = os.path.join(self.__root_save_dir, str(self.__launch_time))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir

    def run(self, ):
        while True:
            if self.__error_try >= self.__tolerance:
                self.__logger.info('%d tfrecord(s) are written into tstamp=%d, time=%s' % (
                    self.__tfrecord_cnt, self.__launch_tstamp, self.__launch_time))
                self.__write_success()
                self.__update_status()
                self.__create_writer()
            tstamp, example = self.__tfrecord_queue.get() 
            self.__writer.write(example)
            self.__tfrecord_cnt += 1
            if self.__launch_tstamp + self.__epoch_interval <= tstamp:
                self.__error_try += 1


#---------------------------------------------------------------------------------------------------#
def test_tfrecord_read(prebatch=256):
    def parse_func(example_proto):
        features = tf.io.parse_single_example(example_proto, features={
            "click": tf.io.FixedLenFeature([1 * prebatch], tf.int64),
            "user_id": tf.io.FixedLenFeature([1 * prebatch], tf.int64),
        })
        return features
    def reshape_input(features):
        label = tf.reshape(features['click'], [-1, 1])
        reshaped = dict()
        reshaped['user_id'] = tf.reshape(features['user_id'], [-1, 1])
        return reshaped, label
    print(prebatch)
    tfrecord_files = ['/mnt/weianjun/workspace/work_dir/tfrecord/20201211075729/part-00000']
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP').map(parse_func)
    features = dataset.make_one_shot_iterator().get_next()
    features, label = reshape_input(features)
    run_list = [label, features['user_id']]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res_list = sess.run(run_list)
        for item in res_list:
            print('-'*50)
            print(item)
    print('-'*50)

def test_tfrecord_write():
    out_file = './realtime/test.tfrecord.gz'
    os.remove(out_file)
    label = [1, 2]
    feat1 = [3, 4]
    feat2 = [1,2,3,4,5,6]
    writer = tf.io.TFRecordWriter(out_file, options='GZIP')
    feature = {
        'is_click': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        'feature1': tf.train.Feature(int64_list=tf.train.Int64List(value=feat1)),
        'feature2': tf.train.Feature(int64_list=tf.train.Int64List(value=feat2)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()


#---------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    #test_tfrecord_read()
    #sys.exit(0)
    conf = Conf(os.path.join(os.getcwd(), 'realtime/conf/test.cfg'))
    conf.set('tfrecord_logger_file', os.path.join(conf.get('work_dir'), 'tfrecord.log'))
    logger = create_logger(conf, 'tfrecord_logger')
    instance_queue = Queue(20000)
    tfrecord_queue = Queue(20000)
    launch_time = '20201201000000'
    kwargs = {
        'feature_schema_file': conf.get('feature_schema_file'),
        'root_save_dir': conf.get('root_save_dir'),
        'prebatch': int(conf.get('prebatch')),
        'save_interval_min': int(conf.get('save_interval_min')),
        'starting_tstamp': int(time.mktime(time.strptime(launch_time, '%Y%m%d%H%M%S'))),
        'instance_queue': instance_queue,
        'tfrecord_queue': tfrecord_queue,
        'logger': logger,
        'debug_mode': True,
    }
    instance_file = os.path.join(conf.get('work_dir'), 'instance_queue.txt')
    if os.path.exists(instance_file):
        for line in open(instance_file, mode='r', encoding='utf8'):
            line = line.strip()
            instance_queue.put(json.loads(line))
    print('%d instances found' % instance_queue.qsize())
    os.system('rm -rf %s/*' % (conf.get('root_save_dir')))
    threads = []
    for i in range(0, int(conf.get('tfrecord_maker_thread_num'))):
        threads.append(TfrecordMaker(thread_name='TfMaker-%d' % (i), **kwargs))
    for i in range(0, int(conf.get('tfrecord_saver_thread_num'))):
        threads.append(TfrecordSaver(thread_name='part-%05d' % (i),  **kwargs))
    for t in threads:
        t.start()
    while True:
        logger.info('Watchdog qsize: instance=%d, tfrecord=%d' % (instance_queue.qsize(), tfrecord_queue.qsize()))
        time.sleep(5)


#end
