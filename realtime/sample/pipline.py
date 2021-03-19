#!/usr/bin/env python
# coding=utf-8
# author: weianjun(weiaj@ushareit.com)
# date: 2020.12.01

import os
import sys
sys.path.append(os.getcwd())

import time
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue
import argparse
from realtime.util.conf import *
from realtime.util.exception import *
from realtime.util.log import create_logger
from realtime.sample.loader import RealtimeReader
from realtime.sample.extractor import FeatureExtractor
from realtime.sample.tfrecord import TfrecordMaker, TfrecordSaver

parser = argparse.ArgumentParser(description='Parse realtime pipeline flags.')
parser.add_argument('--conf_file', type=str, required=True, help='The configuration file for a pipline.')
parser.add_argument('--task_type', type=str, required=True, choices=['tfrecords', 'tftrainer'], 
                    help='Specify pipeline type.')
parser.add_argument('--starting_time', type=str, required=False,  default='', 
                    help='Specify starting time of records userd for training, default is current time, e.g. 20201209201030')

tfrecords_group = parser.add_argument_group(title='tfrecords producation options')
tftrainer_group = parser.add_argument_group(title='tftrainer execution options')

tfrecords_group.add_argument('--realtime_queue_size', type=int, required=False, default=1000, 
                             help='queue size for loading realtime records, default 1000.')
tfrecords_group.add_argument('--instance_queue_size', type=int, required=False, default=1000, 
                             help='queue size for extracting instances, default 1000.')
tfrecords_group.add_argument('--tfrecord_queue_size', type=int, required=False, default=500, 
                             help='queue size for making tfrecords, default 500.')
tfrecords_group.add_argument('--debug_mode', type=bool, required=False, default=False, 
                             help='whether to turn on debug mode.')

args = parser.parse_args()


#---------------------------------------------------------------------------------------------------#
class RealPipeline(object):
    def __init__(self, args):
        if not os.path.exists(args.config_file):
            raise FileNotFound(args.config_file)
        self.__conf = Conf(args.config_file)
        work_dir = self.__conf.get('work_dir') 
        if not os.path.exists(self.__conf.get('root_save_dir')):
            raise FileNotFound(self.__conf.get('root_save_dir'))
        if not os.path.exists(self.__conf.get('work_dir')):
            os.makedirs(self.__conf.get('work_dir'))
        self.__conf.set('realtime_logger_file', os.path.join(work_dir, 'loader.log'))
        self.__conf.set('instance_logger_file', os.path.join(work_dir, 'extractor.log'))
        self.__conf.set('tfrecord_logger_file', os.path.join(work_dir, 'tfrecord.log'))
        self.__conf.set('pipeline_logger_file', os.path.join(work_dir, 'pipeline.log'))
        self.__realtime_logger = create_logger(self.__conf, 'realtime_logger')
        self.__instance_logger = create_logger(self.__conf, 'instance_logger')
        self.__tfrecord_logger = create_logger(self.__conf, 'tfrecord_logger')
        self.__pipeline_logger = create_logger(self.__conf, 'pipeline_logger')
        self.__realtime_queue_size = args.realtime_queue_size
        self.__instance_queue_size = args.instance_queue_size
        self.__tfrecord_queue_size = args.tfrecord_queue_size
        if not args.starting_time:
            self.__launch_tstamp = int(time.time())
        else:
            try:
                time_array = time.strptime(args.starting_time, '%Y%m%d%H%M%S')
            except Exception as e:
                raise RuntimeError('--starting_time format error')
            self.__launch_tstamp = int(time.mktime(time_array))
        self.__epoch_interval = int(self.__conf.get('save_interval_min')) * 60
        self.__debug_mode = args.debug_mode

    def __init_module_kwargs(self, module_name):
        if module_name == 'RealtimeReader':
            kwargs = {
                'app_name': self.__conf.get('app_name'),
                'model_name': self.__conf.get('model_name'),
                'kafka_hosts': self.__conf.get('kafka_hosts'),
                'kafka_ports': self.__conf.get('kafka_ports'),
                'kafka_topic': self.__conf.get('kafka_topic'),
                'request_country': self.__conf.get('request_country'),
                'starting_tstamp': self.__launch_tstamp,
                'record_queue': self.__realtime_queue,
                'logger': self.__realtime_logger,
                'debug_mode': self.__debug_mode,
            }
        elif module_name == 'FeatureExtractor':
            kwargs = {
                'extract_feature_config': self.__conf.get('extract_feature_config'),
                'pos_sample_rate': float(self.__conf.get('pos_sample_rate')),
                'neg_sample_rate': float(self.__conf.get('neg_sample_rate')),
                'request_scenes': self.__conf.get('request_scenes'),
                'request_country': self.__conf.get('request_country'),
                'label_key': self.__conf.get('label_key'),
                'pos_rule': self.__conf.get('pos_rule'),
                'neg_rule': self.__conf.get('neg_rule'),
                'record_queue': self.__realtime_queue,
                'instance_queue': self.__instance_queue,
                'logger': self.__instance_logger,
                'debug_mode': self.__debug_mode,
            }
        elif module_name == 'TfrecordMaker':
            kwargs = {
                'feature_schema_file': self.__conf.get('feature_schema_file'),
                'prebatch': int(self.__conf.get('prebatch')),
                'instance_queue': self.__instance_queue,
                'tfrecord_queue': self.__tfrecord_queue,
                'logger': self.__tfrecord_logger,
                'debug_mode': self.__debug_mode,
            }
        elif module_name == 'TfrecordSaver':
            kwargs = {
                'root_save_dir': self.__conf.get('root_save_dir'),
                'save_interval_min': int(self.__conf.get('save_interval_min')),
                'starting_tstamp': self.__launch_tstamp,
                'tfrecord_queue': self.__tfrecord_queue,
                'logger': self.__tfrecord_logger,
                'debug_mode': self.__debug_mode,
            }
        else:
            raise RuntimeError('Module=%s is unknown' % module_name)
        return kwargs

    def run_tfrecords(self):
        self.__realtime_queue = Queue(self.__realtime_queue_size)
        self.__instance_queue = Queue(self.__instance_queue_size)
        self.__tfrecord_queue = Queue(self.__tfrecord_queue_size)
        kwargs_dict = {
            'RealtimeReader':   self.__init_module_kwargs('RealtimeReader'),
            'FeatureExtractor': self.__init_module_kwargs('FeatureExtractor'),
            'TfrecordMaker':    self.__init_module_kwargs('TfrecordMaker'),
            'TfrecordSaver':    self.__init_module_kwargs('TfrecordSaver')
        }
        threads = []
        for i in range(0, int(self.__conf.get('loader_thread_num'))):
            threads.append(RealtimeReader(thread_name='RealtimeReader-%d' % (i), **kwargs_dict['RealtimeReader']))
        for i in range(0, int(self.__conf.get('extractor_thread_num'))):
            threads.append(FeatureExtractor(thread_name='FeatureExtractor-%d' % (i), **kwargs_dict['FeatureExtractor']))
        for i in range(0, int(self.__conf.get('tfrecord_maker_thread_num'))):
            threads.append(TfrecordMaker(thread_name='TfMaker-%d' % (i), **kwargs_dict['TfrecordMaker']))
        for i in range(0, int(self.__conf.get('tfrecord_saver_thread_num'))):
            threads.append(TfrecordSaver(thread_name='part-%05d' % (i),  **kwargs_dict['TfrecordSaver']))
        for t in threads:
            t.start()
        while True:
            self.__pipeline_logger.info('Watchdog qsize: realtime=%d, instance=%d, tfrecord=%d' % (
                self.__realtime_queue.qsize(), self.__instance_queue.qsize(), self.__tfrecord_queue.qsize()))
            time.sleep(1)
    
    def run_tftrainer(self):
        if not os.path.exists(self.__conf.get('tf_trainer_conf_file')):
            raise FileNotFound('tf_trainer_conf_file')
        increment_model_dir = self.__conf.get('increment_model_dir').rstrip('/') + '/'
        if not os.path.exists(increment_model_dir):
            self.__pipeline_logger.info('increment_model_dir=%s does not exists' % (increment_model_dir))
        elif self.__conf.get('base_ckpt_path') is not None:
            base_ckpt_path = self.__conf.get('base_ckpt_path').rstrip('/') + '/'
            exit_code = os.system('obsutil stat %s' % (base_ckpt_path))
            if exit_code:
                self.__pipeline_logger.info('Find base_ckpt=%s failed' % (base_ckpt_path))
            else:
                self.__pipeline_logger.info('Find base_ckpt=%s OK' % (base_ckpt_path))
            cmd = 'obsutil cp -f -r -flat %s %s' % (base_ckpt_path, increment_model_dir)
            exit_code = os.system(cmd)
            if exit_code:
                self.__pipeline_logger.info('Download base_ckpt=%s failed' % (base_ckpt_path))
            else:
                self.__pipeline_logger.info('Download base_ckpt=%s OK' % (base_ckpt_path))
        while True:
            launch_time = time.strftime("%Y%m%d%H%M%S", time.localtime(self.__launch_tstamp))
            tf_trainer_cmd = 'bash ./work/weshare/real_run.sh %s pipeline %s' % (self.__conf.get('tf_trainer_conf_file'), launch_time)
            exit_code = os.system(tf_trainer_cmd)
            if exit_code:
                self.__pipeline_logger.info('Execute tf_trainer failed, launch_time=%s, exit=%d, cmd=[%s]' % (
                    launch_time, exit_code, tf_trainer_cmd))
            else:
                self.__pipeline_logger.info('Execute tf_trainer OK, launch_time=%s, cmd=[%s]' % (launch_time, tf_trainer_cmd))
            self.__launch_tstamp += self.__epoch_interval
            time.sleep(5)



#---------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    pipeline = RealPipeline(args)
    if args.task_type == 'tfrecords':
        pipeline.run_tfrecords()
    elif args.task_type == 'tftrainer':
        pipeline.run_tftrainer()
    else:
        pass

    

#end
