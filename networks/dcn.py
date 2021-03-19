import tensorflow as tf

from layers import DeepLayer, CrossNet
from networks.base_network import Network


class Dcn(Network):
    def _build(self):
        self.deep = DeepLayer(
            [int(i) for i in self._hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        self.cross = CrossNet(self._flags.cross_depth, l2_reg=self._flags.l2)
        super()._build()

    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            x = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            deep_out = self.deep(x)
            cross_out = self.cross(x)
            return tf.concat((deep_out, cross_out), -1)
