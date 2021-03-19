from networks.base_network import Network
from layers import bi_interaction_layer, DeepLayer
import tensorflow as tf


class NeuralFM(Network):

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
            nfm_in = tf.concat(embeddings, 1)
            nfm_out = bi_interaction_layer(nfm_in)
            deep_out = self.deep(nfm_out)
            return deep_out
