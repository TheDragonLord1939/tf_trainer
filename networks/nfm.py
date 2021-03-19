import tensorflow as tf

from layers import bi_interaction_layer, DeepLayer
from networks.base_network import Network


class Nfm(Network):

    def _build(self):
        self.deep1 = DeepLayer(
            [int(i) for i in self._hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        self.deep2 = DeepLayer(
            [int(i) for i in self._flags.hidden.split(',')],
            activation=self._flags.activation,
            l2_reg=self._flags.l2,
            dropout_rate=self._flags.dropout,
            use_bn=self._flags.use_bn
        )
        super()._build()

    def call(self, dense, embeddings, is_training):
        with tf.name_scope(name=self._name):
            deep_in = tf.concat(dense + [tf.squeeze(emb, [1]) for emb in embeddings], -1)
            dense_out = self.deep1(deep_in)
            nfm_in = tf.concat(embeddings, 1)
            nfm = bi_interaction_layer(nfm_in)
            nfm_out = self.deep2(nfm)
            return tf.concat((dense_out, nfm_out), -1)
