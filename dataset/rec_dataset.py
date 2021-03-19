import csv
import os
from functools import partial

import tensorflow as tf


class RecDataset(object):
    def __init__(self, compression_type, label_key, schema, valid_path, train_path='', batch_size=16, prebatch=512, epochs=1,
                 seq_train=False, **kwargs):
        self.compression_type = compression_type
        self._label_key = label_key
        self._train_set = self._get_file_list(train_path)
        self._valid_set = self._get_file_list(valid_path)
        self._batch_size = batch_size
        self._prebatch = prebatch
        self._epochs = epochs
        self._buffer_size = batch_size * 20
        self._seq_train = seq_train
        with open(schema) as f:
            self._schema = list(csv.reader(f))
        self._parse_feature_meta()
        self.debug_labels = list()
        for item in kwargs.get('debug_labels', '').split(','):
            try:
                name, ftype = item.split(':')
            except Exception as e:
                continue
            self.debug_labels.append((name, ftype))
        self._is_predicting = True if kwargs.get('mode', '') == 'predict' else False
        self._label_dtype = kwargs.get('label_dtype', 'float')
        self._label_num = kwargs.get('label_num', 1)

    def _parse_feature_meta(self):
        """
        self._schema is object of schema.tsv which column info is
        [name, type, voc_size, feature_length, ext_info, tower_index]
        """
        self.numerical_list = [i[0] for i in self._schema if i[1].lower() == 'double']

        self.categorical_list = [i[0] for i in self._schema if i[1].lower() == 'int']

        self.sequence_list = [i[0] for i in self._schema if i[1].lower() in ('sequence', 'weight_sequence')]
        self.weight_sequence_list = ['{}_weight'.format(i[0]) for i in self._schema if i[1].lower() == 'weight_sequence']

        self.varlen_list = [i[0] for i in self._schema if i[1].lower() == 'sparse']

        self.vec_dim = {i[0]: int(i[3]) for i in self._schema if i[1].lower() == 'vector'}
        self.vector_list = list(self.vec_dim.keys())

        self.emb_name = {i[0]: i[4] for i in self._schema}
        self.voc_size = {i[4]: int(i[2]) for i in self._schema}
        self.sequence_length = {i[0]: int(i[3]) for i in self._schema if i[1].lower() in ('sequence', 'weight_sequence')}

        self.tower_index = {i[0]: int(i[5]) for i in self._schema if len(i) > 5}

    def _tfrecord_pipeline(self, file_list, is_training=False):
        epochs = self._epochs if is_training else 1
        filenames = tf.data.Dataset.from_tensor_slices(file_list)
        Dataset = partial(tf.data.TFRecordDataset, compression_type=self.compression_type)
        if self._seq_train:
            dataset = filenames.apply(Dataset)
        else:
            dataset = filenames.apply(
                tf.data.experimental.parallel_interleave(Dataset, cycle_length=20, block_length=16)
            )
        
        if is_training and not self._seq_train:
            dataset = dataset.shuffle(buffer_size=self._buffer_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(self._make_tfrecord_parser, self._batch_size))
        dataset = dataset.map(self._de_prebatch_parser)
        dataset = dataset.prefetch(buffer_size=self._buffer_size)

        return dataset

    def _make_tfrecord_parser(self, record):
        sample_specs = dict()
        sample_specs.update(self.feature_spec)
        sample_specs.update(self.label_spec)
        tf.logging.debug("%s parse_single_example: %r" % (self.__class__.__name__, sample_specs.keys()))
        samples = tf.parse_single_example(record, sample_specs)
        return samples

    def _de_prebatch_parser(self, samples):
        features = dict()
        labels = dict()
        tf.logging.debug(samples.keys())
        tf.logging.debug(self.label_spec.keys())
        for name, value in samples.items():
            if name in self.numerical_list + self.categorical_list:
                features[name] = self._de_prebatch_dense(value, 1)
            elif name in self.sequence_list:
                features[name] = self._de_prebatch_dense(value, self.sequence_length[name])
            elif name in self.vector_list:
                features[name] = self._de_prebatch_dense(value, self.vec_dim[name])
            elif name in self.varlen_list:
                features[name] = self._de_prebatch_sparse(value, self.voc_size[name], self._prebatch)
            elif name in [key for key, _ in self.debug_labels]:
                if self._is_predicting:
                    features[name] = self._de_prebatch_dense(value, 1)
            elif name in self.label_spec:
                features[name] = self._de_prebatch_dense(value, 1)
                labels[name]   = self._de_prebatch_dense(value, 1)
            else:
                tf.logging.warn('tensor {} with shape {} is abandoned in de_prebath'.format(name, value.get_shape()))

        tf.logging.debug("Features numerical: {}".format(self.numerical_list))
        tf.logging.debug("Features categorical: {}".format(self.categorical_list))
        tf.logging.debug("Features sequence: {}".format(self.sequence_list))
        tf.logging.debug("Features vector: {}".format(self.vector_list))
        feature_name_list = sorted(features.keys())
        for feature_name in feature_name_list:
            tf.logging.debug("Feature {}: {}".format(feature_name, features[feature_name].get_shape()))
        label_name_list = sorted(labels.keys())
        for label_name in label_name_list:
            tf.logging.debug("Label {}: {}".format(label_name, labels[label_name].get_shape()))
        return features, labels if len(labels) > 1 else list(labels.values())[0]

    @staticmethod
    def _de_prebatch_dense(feature, dim):
        return tf.reshape(feature, [-1, dim])

    @staticmethod
    def _de_prebatch_sparse(feature, dim, prebatch):
        new_indices = feature.values // dim + feature.indices[:, 0] * prebatch
        new_feature = tf.SparseTensor(
            indices=tf.stack([new_indices, feature.values % dim], axis=1),
            values=feature.values % dim,
            dense_shape=[-1, dim])
        return new_feature

    @property
    def feature_spec(self):
        features = dict()
        for name in self.categorical_list:
            features[name] = tf.io.FixedLenFeature([self._prebatch], tf.int64, [0]*self._prebatch)
        for name in self.sequence_list:
            features[name] = tf.io.FixedLenFeature(
                [self._prebatch * self.sequence_length[name]],
                tf.float32 if name+'_weight' in self.weight_sequence_list else tf.int64,
                [0. if name+'_weight' in self.weight_sequence_list else 0]*self._prebatch*self.sequence_length[name]
            )
        for name in self.varlen_list:
            features[name] = tf.io.VarLenFeature(tf.int64)
        for name in self.vector_list:
            features[name] = tf.io.FixedLenFeature([self.vec_dim[name] * self._prebatch], tf.float32,
                                                   [0.]*self.vec_dim[name]*self._prebatch)
        for name in self.numerical_list:
            features[name] = tf.io.FixedLenFeature([self._prebatch], tf.float32, [0.]*self._prebatch)

        if self._is_predicting:
            for name, ftype in self.debug_labels:
                if name not in features:
                    if ftype == 'int':
                        dtype = tf.int64
                    elif ftype == 'float':
                        dtype = tf.float32
                    elif ftype == 'str':
                        dtype = tf.string
                    else:
                        raise ValueError("ftype {} is illegal".format(ftype))
                    features[name] = tf.io.FixedLenFeature([self._prebatch], dtype)

        return features

    @property
    def label_spec(self):
        labels = dict()
        labels[self._label_key] = tf.FixedLenFeature(
            shape=(self._prebatch * self._label_num,),
            dtype=tf.float32 if self._label_dtype == 'float' else tf.int64,
            default_value=[0. if self._label_dtype == 'float' else 0] * self._prebatch
        )
        return labels

    @staticmethod
    def _get_file_list(path):
        if os.path.isfile(path):
            with open(path) as f:
                return [line.strip() for line in f.readlines()]
        elif os.path.isdir(path):
            files = os.listdir(path)
            return [os.path.join(path, file) for file in files]
        else:
            return None

    def train_set(self):
        return self._tfrecord_pipeline(self._train_set, True)

    def eval_set(self):
        return self._tfrecord_pipeline(self._valid_set)
