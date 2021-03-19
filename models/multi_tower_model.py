import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from models.dssm_model import DssmModel

"""
多塔模型:
labels必须是正样本>0，负样本等于0
"""

class MultiTowerModel(DssmModel):
    def __init__(self):
        super().__init__()
        self.sample_num = self._flags.label_num

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)

            # 左塔
            left_tower    = self._Network(self._flags, self._dataset, 'left_tower')                             # 初始化网络对象
            self.left_features = self._get_features_by_index(features, 0)
            left_dense, left_embeddings = left_tower.build_features(self.left_features)                         # 特征embedding化，[batch_size * pre_batch, length]
            tf.logging.debug("LeftTower input dense: %r" % left_dense)
            tf.logging.debug("LeftTower input embed: %r" % left_embeddings)
            left_tower_out              = left_tower(left_dense, left_embeddings, is_training)                  # 输出
            left_tower_embedding        = tf.nn.l2_normalize(
                tf.keras.layers.Dense(self._flags.vector_dim)(left_tower_out), axis=-1, name=self.left_output)  # 单位化，[batch_size * pre_batch, embedding_size]
            tf.logging.debug("LeftTower output: %r" % left_tower_embedding.shape)

            # 右塔
            right_tower    = self._Network(self._flags, self._dataset, 'right_tower')                           # 初始化网络对象
            self.right_features = self._get_features_by_index(features, 1)
            right_dense, right_embeddings = right_tower.build_features(self.right_features)                     # 特征embedding化，[batch_size * pre_batch * sample_num, length]
            tf.logging.debug("RightTower input dense: %r" % right_dense)
            tf.logging.debug("RightTower input embed: %r" % right_embeddings)
            right_tower_out               = right_tower(right_dense, right_embeddings, is_training)             # 输出
            right_tower_embedding         = tf.nn.l2_normalize(
                tf.keras.layers.Dense(self._flags.vector_dim)(right_tower_out), axis=-1, name=self.right_output)# 单位化，[batch_size * pre_batch * sample_num, embedding_size]
            tf.logging.debug("RightTower output: %r" % right_tower_embedding.shape)

            if mode == estimator.ModeKeys.PREDICT:
                outputs = self._build_predict_outputs(features, right_tower_embedding)
                return estimator.EstimatorSpec(mode, predictions=outputs)

            # 损失
            if self.sample_num > 1:     # 如果是多个sample，那么需要做reshape，使用softmax做分类
                left_tower_embedding  = tf.expand_dims(left_tower_embedding, 1)                                                         # [batch_size * pre_batch, 1, embedding_size]
                tf.logging.debug("LeftTower embedding: %r" % left_tower_embedding.shape)
                right_tower_embedding = tf.reshape(right_tower_embedding, [-1, self.sample_num, right_tower_embedding.shape[-1]])       # [batch_size * pre_batch, sample_num, embedding_size]
                tf.logging.debug("RightTower embedding: %r" % right_tower_embedding.shape)
                cosine_score          = tf.matmul(left_tower_embedding, right_tower_embedding, transpose_b=True)                        # [batch_size * pre_batch, 1, sample_num]
                cosine_score          = tf.squeeze(cosine_score, axis=1, name=self.predictions)                                         # [batch_size * pre_batch, sample_num]
                tf.logging.debug("Cosine: %r" % cosine_score.shape)
                labels                = tf.reshape(labels, [-1, self.sample_num])                                                       # [batch_size * pre_batch, sample_num]
                tf.logging.debug("Labels: %r" % labels.shape)
                loss                  = self.build_loss(labels, cosine_score)
                metrics               = self.build_softmax_metric(labels, cosine_score)
            else:                       # 如果是单个sample，那么使用logloss
                cosine_score = tf.divide(
                    1.+tf.reduce_sum(tf.multiply(left_tower_embedding, right_tower_embedding), axis=-1, keep_dims=True),
                    2., name=self.predictions
                )
                loss         = self._build_loss(labels, cosine_score)
                metrics      = self._build_metrics(labels, cosine_score)

            self._build_summary(loss, metrics)
            if mode == estimator.ModeKeys.EVAL:
                return estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            assert mode == estimator.ModeKeys.TRAIN
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn

    def build_loss(self, labels, logits):
        return tf.losses.softmax_cross_entropy(labels, logits)

    # softmax评估指标，主要关注acc
    def build_softmax_metric(self, labels, logits):
        softmax_prob  = tf.nn.softmax(logits)           # 输入的余弦相似度，需要转成softmax概率
        label_one_hot = tf.cast(labels > 0, tf.int32)   # [1.5, 0.0, 0.0] -> [1, 0, 0]
        with tf.variable_scope('metrics'):
            metrics = {}
            metrics['roc_auc'] = tf.metrics.auc(label_one_hot, softmax_prob)
            metrics['pr_auc']  = tf.metrics.auc(label_one_hot, softmax_prob, curve='PR', summation_method='careful_interpolation')
            metrics["acc"]     = tf.metrics.accuracy(tf.argmax(label_one_hot, 1), tf.argmax(softmax_prob, 1))   # 找到概率最大的index
            return metrics
