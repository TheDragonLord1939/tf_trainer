import os
import shutil
import time

import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib as estimator

from models.base_model import BaseModel

"""
youtubeDNN模型：
    将推荐问题视为一个极多分类问题，使用用户历史播放过的item预测用户下一个播放的item
    具体做法是，特征拼接后过神经网络得到user embedding，然后过item的softmax层，得到分类概率，这样softmax层的权重就是item的embedding

有多种方式可以实现
（1）生成样本的时候，生成item_id -> pv文件，然后使用tf.nn.fixed_unigram_candidate_sampler来做负采样，然后再将采样后的item_id映射成index
（2）生成样本的时候，直接负采样出负样本

label是一个列表，第一个元素是正样本，其后的元素都是负样本
"""

class YoutubeDNNModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.label_num = self._flags.label_num
        self.item_num  = 5000000 

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params, config):
            is_training = True if mode == estimator.ModeKeys.TRAIN else False
            if mode != estimator.ModeKeys.PREDICT:
                features = self.sparse2dense(features, self._dataset.varlen_list)
            features = self._dense2sparse(features, self._dataset.varlen_list)

            # 网络
            network = self._Network(self._flags, self._dataset, 'dnn')                  # 初始化网络对象
            dense_feat, embedding_feat = network.build_features(features)               # 特征embedding化，[batch_size, length]
            tf.logging.debug("feature dense: %r" % dense_feat)
            tf.logging.debug("feature embed: %r" % embedding_feat)
            network_out = network(dense_feat, embedding_feat, is_training)           # 模型输出
            tf.logging.debug("network out: %r" % network_out.shape)

            # user embedding
            user_embedding = tf.keras.layers.Dense(self._flags.vector_dim)(network_out)
            tf.logging.debug("user embedding: %r" % user_embedding.shape)

            if mode == estimator.ModeKeys.PREDICT:
                predictions = {"user_embedding": user_embedding}
                return estimator.EstimatorSpec(mode, predictions=predictions)

            # softmax层
            nce_weights = tf.Variable(tf.truncated_normal([self.item_num, self._flags.vector_dim], stddev = 1.0 / tf.math.sqrt(float(self._flags.vector_dim))), name = "nce_weights")
            logits = tf.matmul(user_embedding, tf.transpose(nce_weights))                               # 计算所有item的概率，[batch_size, item_num]
            labels = tf.reshape(labels, [-1, self.label_num])                                           # [batch_size, label_num]

            if mode == tf.estimator.ModeKeys.EVAL:
                eval_loss = self.build_eval_loss(logits, labels)
                metrics   = self.build_metrics(logits, labels)
                return tf.estimator.EstimatorSpec(mode, loss=eval_loss, eval_metric_ops=metrics)

            assert mode == estimator.ModeKeys.TRAIN
            loss = self.build_loss(user_embedding, labels, nce_weights)                                 # 训练使用nce_loss
            train_op = self._build_train_op(loss)
            return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        return model_fn


    # 训练模式的损失函数，使用nce_loss
    def build_loss(self, inputs, labels, nce_weight):
        inputs = tf.expand_dims(inputs, 1)                                                  # [batch_size, input_dim] -> [batch_size, 1, input_dim]

        weight = tf.nn.embedding_lookup(nce_weight, labels)                                 # [batch_size, label_num, emb_dim]
        logits = tf.matmul(inputs, weight, transpose_b=True)                                # [batch_size, 1, label_num]
        logits = tf.squeeze(logits, [-2])                                                   # [batch_size, label_num]

        # 第一个是正样本，之后是负样本
        label_mask    = tf.constant([1.0] + [0.0] * (self.label_num - 1))                   # label_num, like, [1, 0, 0, ...]
        label_one_hot = tf.multiply(tf.ones_like(logits), label_mask)                       # [batch_size, label_num]

        # nce loss是正样本和负样本贡献之和，然后求均值，参见官方word2vec源码
        xent = tf.nn.sigmoid_cross_entropy_with_logits(labels = label_one_hot, logits = logits)
        loss = tf.reduce_mean(xent)
        return loss

    
    # 评估模式的loss，使用交叉熵
    def build_eval_loss(self, logits, labels):
        # 提取第一列，即正样本
        pos_labels     = tf.gather(labels, 0, axis=1)                                     # [batch_size]
        # 正样本转成one-hot
        labels_one_hot = tf.one_hot(pos_labels, self.item_num)                              # [batch_size, item_num]
        loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits)                      # softmax交叉熵
        return loss


    def build_metrics(self, logits, labels):
        # 提取第一列，即正样本
        pos_labels = tf.gather(labels, 0, axis=1)                                         # [batch_size]
        metrics = {}
        for k in [20, 50]:
            metrics["recall@" + str(k)] = tf.metrics.recall_at_k(labels, logits, int(k))
            metrics["AP@" + str(k)]     = tf.metrics.average_precision_at_k(labels, logits, int(k))
            #correct = tf.nn.in_top_k(logits, tf.squeeze(labels), int(k))
            #metrics["accuary@" + str(k)] = tf.metrics.accuracy(labels=tf.ones_like(labels, dtype=tf.float32), predictions=tf.to_float(correct))
        return metrics

