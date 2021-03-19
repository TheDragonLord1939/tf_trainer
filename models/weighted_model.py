import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from dataset import WeightedDataset, build_dataset
from models.base_model import BaseModel


class WeightedModel(BaseModel):
    def _build_dataset(self):
        dataset = build_dataset(self._flags.dataset)
        if dataset != WeightedDataset:
            raise ValueError('Weighted LR model should be run with WeightedDataset, find {}'.format(dataset.__name__))
        return dataset

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self._parse_sequence_weight(features)
                features = self.sparse2dense(features, self._dataset.varlen_list)
                duration, labels = labels
            features = self._dense2sparse(features, self._dataset.varlen_list)
            network = self._Network(self._flags, self._dataset, 'network')
            dense, embeddings = network.build_features(features)
            network_out = network(dense, embeddings, is_training)
            predictions = tf.keras.layers.Dense(1, activation=tf.sigmoid, name='output')(network_out)
            if mode == estimator.ModeKeys.PREDICT:
                outputs = self._build_predict_outputs(features, predictions)
                return estimator.EstimatorSpec(mode, predictions=outputs)
            metrics = self._build_duration_metrics(duration, labels, predictions)
            weights = self._build_weights(duration, labels)
            loss = self._build_loss(labels, predictions, weights=weights)
            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn

    @staticmethod
    def _build_weights(duration, labels):
        weights = duration / 20. * labels + 1.
        return weights

    def _build_duration_metrics(self, duration, labels, predictions):
        metrics = super()._build_metrics(labels, predictions)
        target = tf.squeeze(predictions, axis=[-1])
        _, indices = tf.nn.top_k(target, k=self._flags.batch_size, sorted=False)
        dtm = tf.reduce_mean(tf.gather(duration, indices), axis=0)
        rdtm = tf.metrics.mean(dtm)
        metrics['rdtm'] = rdtm
        return metrics
