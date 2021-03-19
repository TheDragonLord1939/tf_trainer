from functools import reduce

import tensorflow as tf
from tensorflow_estimator import estimator

from models.mmoe_model import MmoeModel
from keras import backend as K


class MmoeAdaptationModel(MmoeModel):

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self._parse_sequence_weight(features)
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            network = self._Network(self._flags, self._dataset, 'input_layer')
            dense, embeddings = network.build_features(features)
            experts_out = tf.stack(
                [self._Network(self._flags, self._dataset, 'expert_{}'.format(i))(dense, embeddings, is_training)
                 for i in range(self._flags.num_experts)], axis=1
            )
            # 对每个任务都指定一个变量, 在计算loss的时候用上, 实现个性化的loss计算方式
            # 论文代码的地址, https://blog.csdn.net/cdknight_happy/article/details/102618883 
            gates = self._build_gates((dense, embeddings))
            predictions = self._build_predictions(experts_out, gates)
            if mode == estimator.ModeKeys.PREDICT:
                return estimator.EstimatorSpec(mode, predictions=predictions)
            losses = self._build_losses(labels, predictions)
            metrics = self._build_muti_task_metrics(losses, labels, predictions)
            self._build_summary(losses, metrics)
            log_vars = self._build_log_vars(self._dataset.task_types.keys())
            losses = self._build_var_losses(losses, log_vars)
            loss = reduce(tf.add, losses.values())
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn

    def _build_gates(self, inputs):
        """
        :param dense: list dense features of tf.Tensor shaped B x n
        :param embeddings: list embeddings of tf.Tensor shaped B x 1 x emb_dim
        :return gates: dict of expert gates of tf.Tensor shaped B x F x 1 which F is num of experts
        """
        # 只把Relu层应用在gates网络上
        dense, embeddings = inputs
        dense = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
        inputs = tf.keras.layers.Dense(dense.get_shape().as_list()[-1], activation=tf.nn.relu)(dense)
        gates = dict()
        for task_name in self._dataset.task_types.keys():
            gate = tf.keras.layers.Dense(self._flags.num_experts, activation=tf.nn.softmax)(inputs)
            gates[task_name] = tf.expand_dims(gate, axis=-1)
        return gates

    def _build_log_vars(self, task_names):
        log_vars = dict()
        for task_name in task_names:
            log_vars[task_name] = tf.Variable(0., name="log_vars_{}".format(task_name), trainable=True)
        return log_vars

    def _build_var_losses(self, losses, log_vars):
        _losses = dict()
        for task_name, task_type in self._dataset.task_types.items():
            precision = K.exp(-log_vars[task_name])
            if task_type == 'regression':
                _losses[task_name] = precision * losses[task_name] + log_vars[task_name]
            else:
                _losses[task_name] = 2.0 * precision * losses[task_name] + log_vars[task_name]
        return _losses
