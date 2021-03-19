import os
import shutil
import time

import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from models.base_model import BaseModel
from networks import network_factory

class PalModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.left_output = 'bias_output'
        self.right_output = 'predict_output'
        self.left_features = dict()
        self.right_features = dict()
        self.predictions = 'output'

    
    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            self.right_features = self._get_features_by_index(features, 1)
            predict_tower = self._Network(self._flags, self._dataset, 'predict')
            p_dense, p_embeddings = predict_tower.build_features(self.right_features)
            p_tower_out = predict_tower(p_dense, p_embeddings, is_training)
            p_out = tf.keras.layers.Dense(1, activation=tf.sigmoid, name = 'p_out')(p_tower_out)
            if mode == estimator.ModeKeys.PREDICT:
                outputs = self._build_predict_outputs(features, p_out)
                return estimator.EstimatorSpec(
                    mode,
                    predictions=outputs
                )
            self.left_features = self._get_features_by_index(features, 0)
            bias_net = network_factory(self._flags.pal_submodel)
            bias_tower = bias_net(self._flags, self._dataset, 'bias', hidden=self._flags.pal_bias)
            b_dense, b_embeddings = bias_tower.build_features(self.left_features)
            b_tower_out = bias_tower(b_dense, b_embeddings, is_training)
            b_out = tf.keras.layers.Dense(1, activation=tf.sigmoid, name = 'b_out')(b_tower_out)
            predictions = tf.multiply(b_out, p_out, name=self.predictions)
            metrics = self._build_metrics(labels, predictions)
            loss = self._build_loss(labels, predictions)
            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn
    
    def _serving_features(self):
        all_features = super()._serving_features()
        features = {name: value for name, value in all_features.items() if self._dataset.tower_index.get(name, None) == 1}
        return features

    def _get_features_by_index(self, features, index):
        return {name: value for name, value in features.items() if self._dataset.tower_index.get(name, None) == index}

