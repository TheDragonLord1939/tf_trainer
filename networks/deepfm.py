import tensorflow as tf

from layers import fm_net, DeepLayer
from networks.base_network import Network


class DeepFM(Network):
    def _build(self):
        self.deep = DeepLayer(
            [int(i) for i in self._hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        super()._build()

    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            deep_in = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            dense_out = self.deep(deep_in)
            fm_in = tf.concat(embeddings, 1)
            fm_out = fm_net(fm_in)
            return tf.concat((dense_out, fm_out), -1)
