import os
from functools import partial

import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator
from tensorflow_serving.apis import predict_pb2, prediction_log_pb2

from dataset import build_dataset
from networks import network_factory
from models.base_model import BaseModel

class Deepfm_diff_lr_user_id_item_bias(BaseModel):
    
    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self._parse_sequence_weight(features)
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)
            network = self._Network(self._flags, self._dataset, 'network')
            dense, embeddings, linear_weight = network.build_features(features)
            network(dense, embeddings,is_training)
            network_out = network._do(dense, embeddings, linear_weight,is_training)
            tf.logging.warn(network_out)
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

    def _build_train_op(self, loss):
        var_list = tf.trainable_variables("^(?!embedding).*$") if self._flags.freeze_embeddings else None
        optimizer = tf.train.AdamOptimizer(learning_rate=self._flags.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=var_list)
        return train_op
        ####var_list = tf.trainable_variables("^(?!embedding).*$") if self._flags.freeze_embeddings else None
        ###var_list = tf.trainable_variables()
        ###optimizer = tf.train.AdamOptimizer(learning_rate=self._flags.learning_rate)
        ###tf.logging.warn(str(var_list))
        ###tf.logging.warn(str([x.name for x in var_list if "embedding_user_id" not in x.name  ]))
       #### train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=[ x for x in var_list if "embedding_user_id" not in x.name] )
        ###user_id_optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        ###item_id_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        ###tf.logging.warn(str([x for x in var_list if "embedding_user_id" in x.name]))
        ###user_id_embeding = [x for x in var_list if "embedding_user_id" in x.name]
        ###item_id_embeding = [x for x in var_list if "embedding_item_id" in x.name]
        ###rest_var = [x for x in var_list if not ("embedding_user_id"  in x.name or "embedding_item_id" in x.name)  ]
        ###grads = tf.gradients(loss, user_id_embeding + item_id_embeding + rest_var)
        #### user_id_train_op = user_id_optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=[x for x in var_list if "embedding_user_id" in x.name])
        ###
        ###g = tf.abs(grads[0].values)
        ###m_v = tf.nn.moments(g, axes=0)
        ###mean = m_v[0]
        ###var = m_v[1]
        ###for i in range(0, 8, 1):
        ###    tf.summary.scalar("grad_user_mean_{}".format(i), mean[i])
        ###    tf.summary.scalar("grad_user_var_{}".format(i), var[i])
        ####g = tf.abs(grads[:len(user_id_embeding)])
        ####tf.logging.warn("ggggg") 
        ####tf.logging.warn(grads[:1][0].values)
        ####tf.reduce_mean(g, axis=1)
        ###user_id_train_op = user_id_optimizer.apply_gradients(zip(grads[:len(user_id_embeding)],user_id_embeding))
        ###item_id_train_op = item_id_optimizer.apply_gradients(zip(grads[len(user_id_embeding):len(user_id_embeding + item_id_embeding)],item_id_embeding))
        ###train_op = optimizer.apply_gradients(zip(grads[len(user_id_embeding+item_id_embeding):], rest_var),global_step=tf.train.get_global_step() )
        ###train_op_ = tf.group(train_op,user_id_train_op, item_id_train_op) 
        ###return train_op_

