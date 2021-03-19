import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from .mmoe_model import MmoeModel


class RegressorModel(MmoeModel):

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
            # 防止回归值小于0
            predictions = tf.maximum(
                tf.keras.layers.Dense(1, activation=None, name='output')(network_out),
                0.
            )
            if mode == estimator.ModeKeys.PREDICT:
                outputs = {
                    "predictions": predictions,
                    self._flags.label_key: tf.expm1(predictions)
                }
                self._output_cols = list(outputs.keys())
                return estimator.EstimatorSpec(mode, predictions=outputs)
            loss = self._build_regression_loss(labels, predictions)
            metrics = self._build_regression_metrics(loss, labels, predictions, self._flags.label_key)
            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn

    def _build_losses(self, labels, predictions):
        return self._build_regression_loss(labels, predictions)
