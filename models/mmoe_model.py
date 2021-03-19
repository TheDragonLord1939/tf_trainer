from functools import reduce

import tensorflow as tf
from tensorflow_estimator import estimator

from dataset import DurationDataset, MutiTaskDataset, build_dataset
from models.base_model import BaseModel


class MmoeModel(BaseModel):
    def _build_dataset(self):
        dataset = build_dataset(self._flags.dataset)
        if dataset not in (DurationDataset, MutiTaskDataset):
            raise ValueError('mmoe model should be run with DurationDataset or MutiTaskDataset, find {}'
                             .format(dataset.__name__))
        return dataset

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self._parse_sequence_weight(features)
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            network = self._Network(self._flags, self._dataset, 'input_layer')
            dense, embeddings = network.build_features(features)
            # 对齐MMOE的论文的模型
            dense = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            dense = tf.keras.layers.Dense(dense.get_shape().as_list()[-1], activation=tf.nn.relu)(dense)
            assert self._flags.network == "dnn", "If use MMOE model, expert’s network type should be dnn"
            experts_out = tf.stack(
                [self._Network(self._flags, self._dataset, 'expert_{}'.format(i))(dense, None, is_training)
                 for i in range(self._flags.num_experts)], axis=1
            )
            gates = self._build_gates(dense)
            predictions = self._build_predictions(experts_out, gates)
            if mode == estimator.ModeKeys.PREDICT:
                return estimator.EstimatorSpec(mode, predictions=predictions)
            losses = self._build_losses(labels, predictions)
            metrics = self._build_muti_task_metrics(losses, labels, predictions)
            self._build_summary(losses, metrics)
            loss = reduce(tf.add, losses.values())
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn

    def _build_gates(self, inputs):
        """
        :param inputs: list dense features of tf.Tensor shaped B x n
        :return gates: dict of expert gates of tf.Tensor shaped B x F x 1 which F is num of experts
        """
        gates = dict()
        for task_name in self._dataset.task_types.keys():
            gate = tf.keras.layers.Dense(self._flags.num_experts, activation=tf.nn.softmax)(inputs)
            gates[task_name] = tf.expand_dims(gate, axis=-1)
        return gates

    def _build_predictions(self, experts_out, gates):
        """
        :param experts_out: Tensor shaped B x F x D which F is num of experts
        :param gates: dict of expert gates of tf.Tensor shaped B x F x 1 which F is num of experts
        :return predictions: dict of prediction shaped B x 1
        """
        predictions = dict()
        for task_name, task_type in self._dataset.task_types.items():
            gated_out = tf.reduce_sum(experts_out * gates[task_name], axis=1, keepdims=False)
            predictions[task_name] = tf.keras.layers.Dense(
                1, activation=tf.nn.sigmoid if task_type == 'classification' else None,
                name='output_{}'.format(task_name)
            )(gated_out)
        return predictions

    def _build_losses(self, labels, predictions):
        losses = dict()
        for task_name, task_type in self._dataset.task_types.items():
            if task_type == 'regression':
                losses[task_name] = self._build_regression_loss(labels[task_name], predictions[task_name])
            else:
                losses[task_name] = super()._build_loss(labels[task_name], predictions[task_name])
        return losses

    @staticmethod
    def _build_regression_loss(labels, predictions, weights=1.):
        with tf.variable_scope('loss'):
            loss = tf.losses.mean_squared_error(labels, predictions, weights=weights)
        return loss

    def _build_muti_task_metrics(self, losses, labels, predictions):
        metrics = dict()
        for name, task_type in self._dataset.task_types.items():
            if task_type == 'regression':
                metrics.update(self._build_regression_metrics(losses[name], labels[name], predictions[name], name))
            else:
                metrics.update(self._build_classification_metrics(losses[name], labels[name], predictions[name], name))
        return metrics

    @staticmethod
    def _build_regression_metrics(loss, labels, predictions, name):
        with tf.variable_scope('metrics'):
            metrics = dict()
            metrics['{}/loss'.format(name)] = tf.metrics.mean(loss)
            metrics['{}/rmse'.format(name)] = tf.metrics.root_mean_squared_error(labels, predictions)
            metrics['{}/score'.format(name)] = tf.metrics.mean(predictions)
            metrics['{}/label'.format(name)] = tf.metrics.mean(labels)
        return metrics

    @staticmethod
    def _build_classification_metrics(loss, labels, predictions, name):
        with tf.variable_scope('metrics'):
            metrics = dict()
            metrics['{}/loss'.format(name)] = tf.metrics.mean(loss)
            metrics['{}/roc_auc'.format(name)] = tf.metrics.auc(labels, predictions)
            metrics['{}/pr_auc'.format(name)] = tf.metrics.auc(labels, predictions, curve='PR',
                                                               summation_method='careful_interpolation')
            metrics['{}/score'.format(name)] = tf.metrics.mean(predictions)
            metrics['{}/ctr'.format(name)] = tf.metrics.mean(labels)
        return metrics

    @staticmethod
    def _build_summary(losses, metrics):
        for name, metric in metrics.items():
            tf.summary.scalar('{}/training'.format(name), metric[1])
