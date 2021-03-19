import os
from functools import partial

import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator
from tensorflow_serving.apis import predict_pb2, prediction_log_pb2

from dataset import build_dataset
from networks import network_factory


class BaseModel(object):
    def __init__(self):
        self._flags = tf.flags.FLAGS
        tf.logging.warn('tensorflow flags:')
        for key, value in sorted(self._flags.flag_values_dict().items()):
            tf.logging.warn('{:25}: {}'.format(key, value))

        self.dataset_args = {
            'duration_thresholds': self._flags.duration_thresholds, 
            'weight_key': self._flags.weight_key,
            'debug_labels': self._flags.debug_labels,
            'use_sequence_weight': self._flags.use_sequence_weight,
            'label_dtype': self._flags.label_dtype,
            'mode': self._flags.mode,
            'label_num': self._flags.label_num,
            'multi_task_config': self._flags.multi_task_config,
            'negative_rate': self._flags.negative_rate
        }
        self.Dataset = build_dataset(self._flags.dataset)
        self._dataset = self.Dataset(self._flags.compression_type,
                                     self._flags.label_key,
                                     self._flags.schema,
                                     self._flags.valid_path,
                                     self._flags.train_path,
                                     batch_size=self._flags.batch_size,
                                     prebatch=self._flags.prebatch,
                                     epochs=self._flags.epochs,
                                     seq_train=self._flags.seq_train,
                                     **self.dataset_args)
        self._Network = network_factory(self._flags.network)
        self._features = self._serving_features()
        self._build_estimator()
        self._assets = self._serving_warm_up()
        self._output_cols = list()

    def _build_predict_outputs(self, features, predictions):
        outputs = dict()
        if self._flags.mode == 'predict':
            self._output_cols.append('label')
            outputs['label'] = features[self._flags.label_key]
            for name, _ in self._dataset.debug_labels:
                if name in features:
                    outputs[name] = features[name]
                    self._output_cols.append(name)
        outputs[self._flags.label_key] = predictions
        outputs['predictions'] = predictions
        self._output_cols.append(self._flags.label_key)
        return outputs

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self._parse_sequence_weight(features)
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            network = self._Network(self._flags, self._dataset, 'network')
            dense, embeddings = network.build_features(features)
            network_out = network(dense, embeddings, is_training)
            predictions = tf.keras.layers.Dense(1, activation=tf.sigmoid)(network_out)
            if mode == estimator.ModeKeys.PREDICT:
                outputs = self._build_predict_outputs(features, predictions)
                return estimator.EstimatorSpec(mode, predictions=outputs)
            metrics = self._build_metrics(labels, predictions)
            loss = self._build_loss(labels, predictions)
            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn

    @staticmethod
    def sparse2dense(features, sparse_list):
        _features = dict()
        for name, tensor in features.items():
            if name in sparse_list:
                _features[name + '_SparseIndices'] = tensor.indices
                _features[name + '_SparseValues'] = tensor.values
                _features[name + '_SparseShape'] = tensor.dense_shape
            else:
                _features[name] = tensor
        return _features

    @staticmethod
    def _dense2sparse(features, sparse_list):
        _features = dict()
        for name in sparse_list:
            if name + '_SparseIndices' in features:
                _features[name] = tf.SparseTensor(indices=features[name + '_SparseIndices'],
                                                  values=features[name + '_SparseValues'],
                                                  dense_shape=features[name + '_SparseShape'])
            else:
                _features[name] = features[name]
        for name, tensor in features.items():
            if name not in sparse_list:
                _features[name] = tensor
        return _features

    def _parse_sequence_weight(self, features):
        for name in self._dataset.weight_sequence_list:
            sequence_name = name.replace('_weight', '')
            weighted_sequence = features[sequence_name]
            features[sequence_name] = tf.cast(weighted_sequence, tf.int64)
            if self._flags.use_sequence_weight:
                features[name] = tf.mod(weighted_sequence, 1)
        return features

    @staticmethod
    def _build_loss(labels, predictions, weights=1.0):
        with tf.variable_scope('loss'):
            loss = tf.losses.log_loss(labels, predictions, weights=weights)
            return loss

    @staticmethod
    def _build_metrics(labels, predictions):
        with tf.variable_scope('metrics'):
            metrics = dict()
            metrics['roc_auc'] = tf.metrics.auc(labels, predictions)
            metrics['pr_auc'] = tf.metrics.auc(labels, predictions, curve='PR',
                                               summation_method='careful_interpolation')
            metrics['score'] = tf.metrics.mean(predictions)
            metrics['ctr'] = tf.metrics.mean(labels)
        return metrics

    @staticmethod
    def _build_summary(loss, metrics):
        streaming_loss = tf.metrics.mean(loss)
        tf.summary.scalar('training/streaming_loss', streaming_loss[1])
        for name, metric in metrics.items():
            tf.summary.scalar('training/{}'.format(name), metric[1])

    def _build_train_op(self, loss):
        var_list = tf.trainable_variables("^(?!embedding).*$") if self._flags.freeze_embeddings else None
        optimizer = tf.train.AdamOptimizer(learning_rate=self._flags.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=var_list)
        return train_op

    def train(self):
        if self._flags.fixed_training_steps:
            self.estimator.train(self._dataset.train_set, steps=self._flags.fixed_training_steps)
            model_dir = self.estimator.export_saved_model(os.path.join(self._flags.checkpoint_dir, 'export', 'best'),
                                                          serving_input_receiver_fn=self._receiver_fn,
                                                          assets_extra=self._assets)
        else:
            estimator.train_and_evaluate(self.estimator, self._train_spec, self._eval_spec)
            parent = os.path.join(self._flags.checkpoint_dir, 'export', 'best')
            model_dir = os.path.join(parent, os.listdir(parent)[0])
        return model_dir

    def test(self):
        self.estimator.evaluate(self._dataset.eval_set)

    def predict(self):
        result = self.estimator.predict(self._dataset.eval_set)
        to_utf = partial(str, encoding='utf8')
        with open(self._flags.pred_path, 'w') as f:
            for prediction in result:
                strout = []
                for col in self._output_cols:
                    if col in prediction:
                        val = prediction[col].tolist()
                    else:
                        val = ['']
                    strout.extend(val)
                f.write('\t'.join([to_utf(i) if isinstance(i, bytes) else str(i) for i in strout]) + '\n')

    def _serving_features(self):
        features = dict()
        for name in self._dataset.numerical_list:
            features[name] = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=name)
        for name in self._dataset.categorical_list:
            features[name] = tf.placeholder(dtype=tf.int64, shape=[None, 1], name=name)
        for name in self._dataset.sequence_list:
            features[name] = tf.placeholder(dtype=tf.int64,
                                            shape=[None, self._dataset.sequence_length[name]],
                                            name=name)
        if self._flags.use_sequence_weight:
            for name in self._dataset.weight_sequence_list:
                features[name] = tf.placeholder(dtype=tf.float32,
                                                shape=[None,
                                                       self._dataset.sequence_length[name.replace('_weight', '')]],
                                                name=name)
        for name in self._dataset.vector_list:
            features[name] = tf.placeholder(dtype=tf.float32, shape=[None, self._dataset.vec_dim[name]], name=name)
        for name in self._dataset.varlen_list:
            features[name + '_SparseIndices'] = tf.placeholder(dtype=tf.int64, shape=[None, 2], name=name)
            features[name + '_SparseValues'] = tf.placeholder(dtype=tf.int64, shape=[None], name=name)
            features[name + '_SparseShape'] = tf.placeholder(dtype=tf.int64, shape=[2], name=name)
        return features

    @property
    def _receiver_fn(self):
        return estimator.export.build_raw_serving_input_receiver_fn(self._features)

    def _build_estimator(self):
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        distribute = tf.distribute.MirroredStrategy() if self._flags.distribute else None
        config = estimator.RunConfig(
            model_dir=self._flags.checkpoint_dir,
            save_summary_steps=self._flags.save_summary_steps,
            save_checkpoints_steps=self._flags.checkpoints_steps,
            keep_checkpoint_max=2,
            session_config=self.sess_config,
            log_step_count_steps=self._flags.log_step_count_steps,
            train_distribute=distribute,
            eval_distribute=distribute
        )

        warm_start_setting = None
        if os.path.exists(self._flags.warm_start_dir):
            vars_to_warm_start = '.*/embeddings[^/]' if self._flags.warm_start_mode == 'emb' else '.*'
            warm_start_setting = estimator.WarmStartSettings(
                ckpt_to_initialize_from=self._flags.warm_start_dir,
                vars_to_warm_start=vars_to_warm_start)

        self.estimator = estimator.Estimator(self._build_model_fn(), config=config, warm_start_from=warm_start_setting)

    def _build_voc_info(self):
        voc_info = dict()
        voc_size = {name: size for name, size in self._dataset.voc_size.items()
                    if name in self._dataset.categorical_list+self._dataset.sequence_list+self._dataset.varlen_list}
        for name, size in voc_size.items():
            voc = tf.train.VocabInfo(
                new_vocab=os.path.join(self._flags.checkpoint_dir, "vocabularies", name),
                new_vocab_size=size,
                num_oov_buckets=0,
                old_vocab=os.path.join(
                    self._flags.warm_start_dir, "vocabularies", name),
                backup_initializer=tf.initializers.random_uniform()
            )
            voc_info["embedding_{}/embeddings".format(name)] = voc
        return voc_info

    def _build_vocabularies(self):
        parent = os.path.join(self._flags.checkpoint_dir, 'vocabularies')
        os.makedirs(parent, exist_ok=True)
        voc_size = {name: size for name, size in self._dataset.voc_size.items()
                    if name in self._dataset.categorical_list+self._dataset.sequence_list+self._dataset.varlen_list}
        for name, size in voc_size.items():
            with open(os.path.join(parent, name), 'w') as f:
                f.write('\n'.join([str(i) for i in range(size)]))

    def _serving_warm_up(self):
        _dataset = self.Dataset(self._flags.compression_type,
                                self._flags.label_key,
                                self._flags.schema,
                                self._flags.valid_path,
                                prebatch=self._flags.prebatch,
                                batch_size=1,
                                **self.dataset_args)
        feature, labels = _dataset.eval_set().make_one_shot_iterator().get_next()
        feature = self._parse_sequence_weight(feature)
        feature = self.sparse2dense(feature, self._dataset.varlen_list)
        feature = {name: tensor for name, tensor in feature.items() if name in self._features}
        with tf.Session(config=self.sess_config) as sess:
            feature_n = sess.run(feature)
        del sess
        del _dataset
        request = predict_pb2.PredictRequest()
        for k, v in feature_n.items():
            request.inputs[k].CopyFrom(tf.make_tensor_proto(v, shape=v.shape))
        log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
        filename = 'tf_serving_warmup_requests'
        file_dir = self._flags.checkpoint_dir
        path = os.path.join(file_dir, filename)
        os.makedirs(file_dir, exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        with tf.python_io.TFRecordWriter(path) as writer:
            writer.write(log.SerializeToString())
        return {filename: path}

    @property
    def _train_spec(self):
        hooks = [tf.contrib.estimator.stop_if_no_decrease_hook(self.estimator, 'loss',
                                                               max_steps_without_decrease=self._flags.patient)]
        return estimator.TrainSpec(input_fn=self._dataset.train_set, hooks=hooks)

    @property
    def _eval_spec(self):
        exporter = tf.estimator.BestExporter('best', self._receiver_fn, exports_to_keep=1, assets_extra=self._assets)
        return estimator.EvalSpec(input_fn=self._dataset.eval_set, steps=None,
                                  exporters=exporter,
                                  throttle_secs=self._flags.eval_throttle)
