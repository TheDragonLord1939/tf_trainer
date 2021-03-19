import os
import shutil
import time

import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from models.base_model import BaseModel


class DssmModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.left_output = 'user_vector'
        self.right_output = 'item_vector'
        self.left_features = dict()
        self.right_features = dict()
        self.predictions = 'output'

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            self.left_features = self._get_features_by_index(features, 0)
            self.right_features = self._get_features_by_index(features, 1)
            user_tower = self._Network(self._flags, self._dataset, 'user')
            item_tower = self._Network(self._flags, self._dataset, 'item')
            u_dense, u_embeddings = user_tower.build_features(self.left_features)
            v_dense, v_embeddings = item_tower.build_features(self.right_features)
            u_tower_out = user_tower(u_dense, u_embeddings, is_training)
            v_tower_out = item_tower(v_dense, v_embeddings, is_training)
            u_vector = tf.nn.l2_normalize(
                tf.keras.layers.Dense(self._flags.vector_dim)(u_tower_out),
                axis=-1, name=self.left_output
            )
            v_vector = tf.nn.l2_normalize(
                tf.keras.layers.Dense(self._flags.vector_dim)(v_tower_out),
                axis=-1, name=self.right_output
            )
            # 使用余弦相似度，计算结果在[-1, 1]，变换成[0, 1]
            predictions = tf.divide(1. + tf.reduce_sum(tf.multiply(u_vector, v_vector), axis=-1, keep_dims=True), 2., name=self.predictions)
            if mode == estimator.ModeKeys.PREDICT:
                if self._flags.predict_with_emb:
                    outputs = self._build_predict_outputs(features, v_vector)
                elif self._flags.predict_with_user:
                    outputs = self._build_predict_outputs_user(features, u_vector)
                else:
                    outputs = super(DssmModel, self)._build_predict_outputs(features, predictions)
                return estimator.EstimatorSpec(
                    mode,
                    predictions=outputs
                )
            metrics = self._build_metrics(labels, predictions)
            loss = self._build_loss(labels, predictions)
            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn


    def _build_predict_outputs_user(self, features, predictions):
        outputs = dict()
        if self._flags.mode == 'predict':
            for name, _ in self._dataset.debug_labels:
                if name in features:
                    outputs[name] = features[name]
                    self._output_cols.append(name)
        outputs['user_embedding'] = predictions
        self._output_cols.append('user_embedding')
        return outputs


    def _get_features_by_index(self, features, index):
        return {name: value for name, value in features.items() if self._dataset.tower_index.get(name, None) == index}

    def _build_predict_outputs(self, features, predictions):
        outputs = dict()
        if self._flags.mode == 'predict':
            for name, _ in self._dataset.debug_labels:
                if name in features:
                    outputs[name] = features[name]
                    self._output_cols.append(name)
        outputs['item_embedding'] = predictions
        self._output_cols.append('item_embedding')
        return outputs

    def train(self):
        model_dir = super().train()
        model_dir = model_dir.decode('utf-8') if isinstance(model_dir, bytes) else model_dir
        del self.estimator
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            modified_dir = os.path.join(self._flags.checkpoint_dir, 'export', 'best', str(int(time.time())))

            graph = sess.graph
            user_inputs = {name: graph.get_tensor_by_name('{}:0'.format(name)) for name in self.left_features.keys()}
            user_outputs = {self.left_output: graph.get_tensor_by_name('{}:0'.format(self.left_output))}
            item_inputs = {name: graph.get_tensor_by_name('{}:0'.format(name)) for name in self.right_features.keys()}
            item_outputs = {self.right_output: graph.get_tensor_by_name('{}:0'.format(self.right_output))}
            predictions = {self.predictions: graph.get_tensor_by_name('{}:0'.format(self.predictions))}

            user_signature = tf.saved_model.predict_signature_def(inputs=user_inputs, outputs=user_outputs)
            item_signature = tf.saved_model.predict_signature_def(inputs=item_inputs, outputs=item_outputs)

            user_inputs.update(item_inputs)
            serve_signature = tf.saved_model.predict_signature_def(inputs=user_inputs, outputs=predictions)

            builder = tf.saved_model.builder.SavedModelBuilder(modified_dir)
            builder.add_meta_graph_and_variables(
                sess, tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: serve_signature,
                    'user_vector': user_signature,
                    'item_vector': item_signature
                }
            )

            builder.save()
            shutil.move(os.path.join(model_dir, 'assets.extra'), modified_dir)
            shutil.rmtree(model_dir)
        return modified_dir
