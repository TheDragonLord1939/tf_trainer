import tensorflow as tf
from networks.base_network import Network


class LR(Network):
    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            return tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
