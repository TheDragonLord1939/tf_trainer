import tensorflow as tf

from layers import CIN, DeepLayer
from networks.base_network import Network


class Cin(Network):

    def _build(self):
        self.deep = DeepLayer(
            hidden_units=[int(i) for i in self._hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        self.cin = CIN(
            layer_size=[int(i) for i in self._flags.cin_layers.split(',')],
            activation=self._flags.activation,
            split_half=self._flags.split_half,
            l2_reg=self._flags.l2)
        super()._build()

    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            deep_in = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            deep_out = self.deep(deep_in)

            cin_in = tf.concat(embeddings, 1)
            cin_out = self.cin(cin_in)
            return tf.concat((deep_out, cin_out), -1)
