import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from models.dssm_model import DssmModel
from models.regressor_model import RegressorModel


class FmRecallModel(DssmModel, RegressorModel):

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            self.left_features = self._get_features_by_index(features, 0)
            self.right_features = self._get_features_by_index(features, 1)
            left_network = self._Network(self._flags, self._dataset, 'left')
            right_network = self._Network(self._flags, self._dataset, 'right')
            u_dense, u_embeddings = left_network.build_features(self.left_features)
            v_dense, v_embeddings = right_network.build_features(self.right_features)
            left_embedding = tf.reduce_mean(tf.concat(u_embeddings, 1), 1)
            right_embedding = tf.reduce_mean(tf.concat(v_embeddings, 1), 1)
            dense = u_dense + v_dense
            embeddings = u_embeddings + v_embeddings
            network_out = left_network(dense, embeddings, is_training)
            predictions = tf.keras.layers.Dense(1, activation=None, name='predictions')(network_out)
            predictions = tf.identity(predictions, name=self.predictions)
            if mode == estimator.ModeKeys.PREDICT:
                left_bias = tf.reduce_mean(tf.ones_like(left_embedding), -1, keepdims=True)
                left_embedding = tf.concat((left_embedding, left_bias), -1, name=self.left_output)
                right_bias = tf.reduce_mean(right_network(v_dense, v_embeddings, False), -1, keepdims=True)
                right_embedding = tf.concat((right_embedding, right_bias), -1, name=self.right_output)
                outputs = self._build_predict_outputs(features, right_embedding)
                return estimator.EstimatorSpec(
                    mode,
                    predictions=outputs
                )
            loss = self._build_regression_loss(labels, predictions)
            metrics = self._build_regression_metrics(loss, labels, predictions, 'duration')
            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn
